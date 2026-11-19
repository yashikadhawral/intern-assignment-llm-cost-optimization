from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import (
    ACTIONS,
    SYSTEM_PROMPT,
    compact_snapshot,
    parse_resolution,
    strip_article,
)


@dataclass
class TicketResolution:
    action: str
    kb_article: str


@dataclass
class Triage:
    """Structured triage result — passed between nodes instead of free text."""

    intent: str  # refund | replace | other
    product_type: str
    flags: list[str] = field(default_factory=list)

    def query_terms(self) -> list[str]:
        return [self.intent, self.product_type, *self.flags]


# ---------------------------------------------------------------------------
# Retrieval index over the help center.
#
# The expanded help center (shared.kb.expanded_help_center_blob, >200k tokens) is
# real articles + internal filler that "does not change customer policy unless an
# article id is cited". We index the cited articles once and never send the blob.
# ---------------------------------------------------------------------------

_SYNONYMS = {
    "refund": ["refund", "refunded", "refundable", "return", "returns"],
    "replace": ["replace", "replacement", "damaged", "damage"],
    "digital": ["digital", "license", "download"],
    "license_revealed": ["license", "revealed"],
    "subscription": ["subscription", "subscriptions", "annual", "cancelled", "prorated"],
    "custom": ["custom", "personalized"],
    "customized": ["custom", "personalized"],
    "damage_reported": ["damaged", "damage", "replacement"],
    "physical": ["physical", "resalable", "delivery"],
    "unopened": ["resalable"],
}

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@lru_cache
def _index() -> tuple[dict[str, Counter], dict[str, float]]:
    docs = {aid: Counter(_tokens(aid.replace("-", " ") + " " + load_article(aid))) for aid in article_ids()}
    n = len(docs)
    df = Counter(t for c in docs.values() for t in c)
    idf = {t: math.log(1 + n / d) for t, d in df.items()}
    return docs, idf


def search_articles(terms: list[str], k: int = 2) -> list[str]:
    docs, idf = _index()
    expanded = [w for t in terms for w in _SYNONYMS.get(t, [t])]
    scores = {
        aid: sum(idf.get(w, 0.0) * min(tf[w], 2) for w in expanded)
        for aid, tf in docs.items()
    }
    ranked = sorted((a for a in scores if scores[a] > 0), key=lambda a: -scores[a])
    return ranked[:k]


class SupportPipeline:
    """
    Same product as Question 1 — triage, retrieve, compose, audit (optimized).

    * triage   – deterministic: order facts + intent keywords → structured Triage
    * retrieve – local lexical search over the indexed KB (top-2 short articles)
    * compose  – the ONLY LLM call: compact prompt, JSON answer
    * audit    – deterministic schema/policy validation; one fallback LLM call
                 (with a wider article set) only if validation fails
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        # Note: the 200k-token help-center blob is intentionally NOT loaded.

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = compact_snapshot(order)

        triage = self._triage(thread, order)
        retrieved = self._retrieve(triage)
        draft = self._compose(thread, snapshot, retrieved)
        final = self._audit(thread, snapshot, triage, retrieved, draft)
        return TicketResolution(final.action, final.kb_article)

    # -- node 1 ------------------------------------------------------------
    def _triage(self, thread: str, order: dict) -> Triage:
        text = thread.lower()
        if any(w in text for w in ("replace", "replacement", "cracked", "broken", "damaged")):
            intent = "replace"
        elif any(w in text for w in ("refund", "money back", "return", "cancel")):
            intent = "refund"
        else:
            intent = "other"

        flags = [f for f in ("license_revealed", "customized", "damage_reported") if order.get(f)]
        if order.get("product_type") == "physical" and not order.get("opened"):
            flags.append("unopened")
        return Triage(intent, str(order.get("product_type", "")), flags)

    # -- node 2 ------------------------------------------------------------
    def _retrieve(self, triage: Triage, k: int = 2) -> list[str]:
        return search_articles(triage.query_terms(), k=k)

    # -- node 3 ------------------------------------------------------------
    def _compose(self, thread: str, snapshot: str, article_ids_: list[str], label: str = "compose"):
        policies = "\n".join(f"[{a}] {strip_article(load_article(a))}" for a in article_ids_)
        messages = [
            ChatMessage("system", SYSTEM_PROMPT),
            ChatMessage(
                "user",
                f"POLICIES:\n{policies}\nTHREAD:\n{thread}\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
            ),
        ]
        return parse_resolution(self.llm.chat(label, messages))

    # -- node 4 ------------------------------------------------------------
    def _audit(self, thread, snapshot, triage: Triage, retrieved: list[str], draft):
        if self._valid(draft, retrieved):
            return draft
        # Rare path: widen retrieval and re-run compose once.
        wider = search_articles(triage.query_terms(), k=5) or article_ids()
        retry = self._compose(thread, snapshot, wider, label="audit_retry")
        if not self._valid(retry, wider):
            raise ValueError(f"Audit failed: {retry.raw_completion[:200]}")
        return retry

    @staticmethod
    def _valid(res, allowed_articles: list[str]) -> bool:
        return res.action in ACTIONS and res.kb_article in allowed_articles


def debug_dump(ticket_id: str) -> str:
    """Helper for the write-up: show what each node produced."""
    p = SupportPipeline()
    ticket = get_ticket(ticket_id)
    order = get_order(ticket["order_id"])
    tri = p._triage(format_thread(ticket), order)
    return json.dumps({"triage": tri.__dict__, "retrieved": p._retrieve(tri)}, indent=2)
