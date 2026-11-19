"""Print per-call token / points breakdown for Q1 and Q2 (optimized agents).

    python scripts/cost_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.billing import CACHED_INPUT_PRICE, INPUT_PRICE, OUTPUT_PRICE  # noqa: E402
from question1.agent import SupportCopilot  # noqa: E402
from question2.agent import SupportPipeline  # noqa: E402

TICKETS = ["T-1001", "T-1002", "T-1003", "T-1004", "T-1091"]


def report(name, factory, target):
    print(f"\n## {name} (target <= {target} pts/ticket)\n")
    print("| Ticket | Call | Input tok | Cached | Output tok | Input pts | Output pts | Points |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    totals = []
    for tid in TICKETS:
        agent = factory()
        res = agent.handle_ticket(tid)
        for c in agent.session.calls:
            cached = min(c.cached_prefix_tokens, c.input_tokens)
            inp = (c.input_tokens - cached) * INPUT_PRICE + cached * CACHED_INPUT_PRICE
            out = c.output_tokens * OUTPUT_PRICE
            print(f"| {tid} | {c.label} | {c.input_tokens} | {cached} | {c.output_tokens} | {inp:.0f} | {out} | {c.points()} |")
        pts = agent.session.total_points()
        totals.append(pts)
        print(f"| **{tid}** | **total** → {res.action} / {res.kb_article} | | | | | | **{pts}** |")
    pub = totals[:4]
    print(f"\nPublic avg: {sum(pub)/len(pub):.0f} pts, max: {max(pub)} pts")


if __name__ == "__main__":
    report("Question 1 — SupportCopilot", SupportCopilot, 3000)
    report("Question 2 — SupportPipeline", SupportPipeline, 2500)
