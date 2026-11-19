from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket


@dataclass
class TicketResolution:
    action: str
    kb_article: str
    customer_reply: str
    raw_completion: str


def parse_resolution(text: str) -> TicketResolution:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    return TicketResolution(
        action=str(data["action"]).upper(),
        kb_article=str(data["kb_article"]),
        customer_reply=str(data.get("customer_reply", "")),
        raw_completion=text,
    )


ACTIONS = ("REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2")

# Order fields the routing decision actually depends on (ids/SKUs add tokens, not signal).
DECISION_FIELDS = (
    "title",
    "product_type",
    "delivered_days_ago",
    "subscription_age_days",
    "opened",
    "license_revealed",
    "customized",
    "damage_reported",
)

# Static, byte-identical system prompt: no run ids, no few-shot, no "reason step by step"
# (those wording triggers make the model emit ~480 output tokens at 15 pts each).
SYSTEM_PROMPT = (
    "You are ShopAssist, a Tier-1 support agent. Using only the POLICIES and ORDER_SNAPSHOT, "
    f"pick one action from {'|'.join(ACTIONS)} and the policy id that backs it. "
    'Answer with JSON only: {"action":"","kb_article":"","customer_reply":"<1 short sentence>"}'
)


def compact_snapshot(order: dict) -> str:
    """Minified JSON with only decision-relevant fields."""
    slim = {k: order[k] for k in DECISION_FIELDS if k in order}
    return json.dumps(slim, separators=(",", ":"))


def strip_article(text: str) -> str:
    """Drop the markdown title line (it duplicates the id) and emphasis markers."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    return " ".join(lines).replace("**", "").strip()


def relevant_article_ids(order: dict, thread: str) -> list[str]:
    """
    Pre-filter the help center to the policies that can apply to this order.

    Uses order facts first (cheap, reliable) and the customer's intent from the
    thread second. Returns at most 2 ids so the model still has a real choice
    (e.g. damage vs. custom-item rule) without paying for unrelated policies.
    """
    text = thread.lower()
    ptype = order.get("product_type")
    picked: list[str] = []

    if ptype == "subscription":
        picked.append("subscription-cooling-off")
    if ptype == "digital" or order.get("license_revealed"):
        picked.append("digital-license-no-refund")
    if order.get("damage_reported") or any(w in text for w in ("damaged", "cracked", "broken", "replacement")):
        if order.get("customized") or ptype == "custom":
            picked.append("custom-items-no-replace")
        picked.append("damage-replacement-7-days")
    if ptype == "physical" and not picked:
        picked.append("returns-within-30-days")

    known = set(article_ids())
    picked = [a for a in dict.fromkeys(picked) if a in known]
    return picked[:2] or ["returns-within-30-days"]


def policies_block(ids: list[str]) -> str:
    return "\n".join(f"[{aid}] {strip_article(load_article(aid))}" for aid in ids)


class SupportCopilot:
    """
    ShopAssist Tier-1 copilot (optimized).

    One LLM call per ticket with: a short static system prompt, only the 1-2
    policies relevant to the order, the thread, and a minified order snapshot.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def build_messages(self, ticket_id: str) -> list[ChatMessage]:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        ids = relevant_article_ids(order, thread)

        user = "\n".join(
            [
                f"POLICIES:\n{policies_block(ids)}",
                f"THREAD:\n{thread}",
                f"ORDER_SNAPSHOT:\n{compact_snapshot(order)}\nEND_SNAPSHOT",
            ]
        )
        return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user)]

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        messages = self.build_messages(ticket_id)
        completion = self.llm.chat("handle_ticket", messages)
        return parse_resolution(completion)
