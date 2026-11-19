# ShopAssist — LLM Cost Optimization: Solution

**Result:** every ticket stays correct (public tickets + holdout `T-1091`), and cost falls well below both targets.

| | Baseline (pts/ticket) | Optimized (pts/ticket) | Target | Reduction |
|---|---:|---:|---:|---:|
| **Q1** single call | ~8,863 (avg) | **1,190** (avg), 1,314 max | ≤ 3,000 | **−87%** |
| **Q2** 4-node pipeline | 73,269 (T-1001) | **1,256** (avg), 1,395 max | ≤ 2,500 | **−98%** |

Billing: `points = uncached_input × 3 + cached_input × 0.3 + output × 15`

```bash
pip install -r requirements.txt
pytest -q                       # 7 tests: correctness + cost targets
python question1/run.py
python question2/run.py
python scripts/cost_report.py   # per-call breakdown tables (below)
```

---

## Where the baseline points went

Output tokens cost **5× input**, so the first thing to check was the output.

1. **Verbose output (biggest Q1 cost).** The baseline prompt says *"Think step by step… quote any article"*. That makes the model write ~480 tokens of reasoning before the JSON: 480 × 15 ≈ **7,200 pts per call**. Nobody reads that reasoning; only `action` and `kb_article` are graded. A JSON-only answer is ~40 tokens (~600 pts).
2. **Sending the whole help center.** Q1 sends all 5 articles plus 4 few-shot examples. Q2 sends up to 120k chars of the 210k-token expanded help center in *triage*, and 80k more in *audit*. Most of it is filler ("Internal routing note… does not change customer policy").
3. **Repeated context in Q2.** Thread + snapshot go to all 4 nodes; triage text is sent twice to compose ("TRIAGE AGAIN"); audit gets draft + triage + articles + help center again.
4. **LLM calls that don't need an LLM.** *Retrieve* asks an LLM to "print" articles we already have on disk. *Audit* asks an LLM to rewrite JSON that code can validate.
5. **Cache-busting noise.** A random `Run id {uuid}` / `trace-{uuid}` at the start of the system prompt means no two prompts share a prefix. (Prompts are now far below the 1,024-token cache minimum, so caching isn't needed. Padding them to reach it would cost more than it saves.)
6. **Pretty-printed order JSON** (`indent=2`) and fields that don't affect the decision (`order_id`, `sku`).

---

## Q1 — `question1/agent.py`

| Change | Why |
|---|---|
| A short, **static** system prompt: role, the 4 allowed actions, a JSON-only output schema | No verbose trigger words, so the output drops from ~480 to ~40 tokens. A byte-identical prompt is also cache-friendly. |
| Removed the few-shot block and the `uuid` run id | Saves tokens, and the prompt no longer changes on every call |
| **Policy pre-filter** `relevant_article_ids(order, thread)`: picks at most 2 articles from order facts (`product_type`, `license_revealed`, `customized`, `damage_reported`) and the customer's intent | Only relevant policies go to the LLM. Sending 2 when it's ambiguous (e.g. damaged **custom** mug → custom rule + damage rule) still lets the model make the call. |
| `strip_article`: drops the markdown header that repeats the id, and the `**` markers | A few tokens per article |
| `compact_snapshot`: minified JSON with only decision fields | About 50% fewer snapshot tokens |
| `parse_resolution` and the `TicketResolution` dataclass unchanged | Graded fields and parsing stay compatible |

### Q1 per-call breakdown (one call per ticket)

| Ticket | Input tok | Output tok | Input pts | Output pts | **Points** | Decision |
|---|---:|---:|---:|---:|---:|---|
| T-1001 | 208 | 46 | 624 | 690 | **1,314** | REFUND / returns-within-30-days |
| T-1002 | 191 | 39 | 573 | 585 | **1,158** | SEND_ARTICLE / digital-license-no-refund |
| T-1003 | 206 | 39 | 618 | 585 | **1,203** | ESCALATE_TIER2 / custom-items-no-replace |
| T-1004 | 181 | 36 | 543 | 540 | **1,083** | REPLACE / damage-replacement-7-days |
| T-1091 (holdout) | 191 | 38 | 573 | 570 | **1,143** | SEND_ARTICLE / subscription-cooling-off |

Baseline for comparison: T-1001 used 526 input + 482 output tokens = 8,808 pts.

---

## Q2 — `question2/agent.py`

The four stages are still there, but only the stage that needs language reasoning uses the LLM. The stages pass **structured data** to each other instead of free text.

```
ticket + order
   │
   ▼
[1 triage]   deterministic → Triage(intent, product_type, flags)         0 pts
   │
   ▼
[2 retrieve] local TF-IDF search over indexed KB articles → top-2 ids    0 pts
   │
   ▼
[3 compose]  ONE LLM call: static system prompt + 2 short policies
             + thread + minified snapshot → JSON                       ~1,250 pts
   │
   ▼
[4 audit]    code validation: action ∈ allowed set, kb_article ∈ retrieved
             └─ only if it fails: one retry with top-5 articles (rare path)
```

| Node | Baseline | Optimized |
|---|---|---|
| **Triage** | LLM call carrying a 500-title index + 120k chars of help center (8,451 in / 482 out = **32,583 pts**) | Rule-based extraction from order facts and thread keywords. Order facts come from `get_order`, so they're already structured. **0 pts** |
| **Retrieve** | LLM asked to "print" articles (**2,460 pts**) | `search_articles()`: TF-IDF index built once (`lru_cache`) over the real KB articles, with a small synonym map (e.g. `customized → custom, personalized`). The 210k-token `expanded_help_center_blob` is **never loaded or sent**: its filler states it doesn't change policy unless an article id is cited. This also removes ~3 min of start-up time. **0 pts** |
| **Compose** | Thread + snapshot + triage ×2 + articles, "full reasoning" (**10,833 pts**) | Reuses Q1's compact prompt with only the top-2 retrieved policies. **~1,250 pts** |
| **Audit** | LLM call carrying 80k chars of help center + everything again (**27,393 pts**) | Deterministic schema/policy check. An LLM retry happens **only** if the draft is invalid, so a guardrail stays without paying for it on every ticket. **0 pts** on the normal path |

Retrieval result (correct article ranked #1 in every case):

| Ticket | Triage | Retrieved (top-2) |
|---|---|---|
| T-1001 | refund · physical · unopened | returns-within-30-days, digital-license-no-refund |
| T-1002 | refund · digital · license_revealed | digital-license-no-refund, returns-within-30-days |
| T-1003 | replace · custom · customized, damage_reported | custom-items-no-replace, damage-replacement-7-days |
| T-1004 | replace · physical · damage_reported | damage-replacement-7-days, custom-items-no-replace |
| T-1091 | refund · subscription | subscription-cooling-off, returns-within-30-days |

### Q2 breakdown — physical fulfillment path (T-1004, REPLACE)

| Node | Input tok | Output tok | Points |
|---|---:|---:|---:|
| triage | — | — | 0 |
| retrieve | — | — | 0 |
| compose | 208 | 36 | 1,164 |
| audit | — | — | 0 (passed) |
| **Total** | | | **1,164** |

### Q2 breakdown — denial/escalation path (T-1003, ESCALATE_TIER2)

| Node | Input tok | Output tok | Points |
|---|---:|---:|---:|
| triage | — | — | 0 |
| retrieve | — | — | 0 |
| compose | 206 | 39 | 1,203 |
| audit | — | — | 0 (passed) |
| **Total** | | | **1,203** |

Other tickets: T-1001 = 1,395 · T-1002 (denial) = 1,263 · T-1091 (holdout denial) = 1,248.

**Baseline T-1001:** triage 32,583 + retrieve 2,460 + compose 10,833 + audit 27,393 = **73,269 pts**.

---

## Tests

`tests/test_question*.py`: the `test_baseline_cost_is_high` checks (which asserted the *old* cost) were replaced with:

- `test_cost_within_target`: each public ticket ≤ 3,000 (Q1) / ≤ 2,500 (Q2), and Q1 uses exactly one call
- `test_holdout_subscription` added for Q2 as well

All the original correctness tests are unchanged. `eval/reference.py` is not imported by either agent.

## Trade-offs and next steps

- **Rule-based triage and pre-filter** are cheap and exact for structured order facts, but they need updating when new policy types are added. For a larger KB, swap `search_articles` for embeddings (still 0 LLM points per ticket after indexing).
- **`customer_reply` is kept but short** (one sentence). If the product needs a longer reply, it can be generated only for actions that reach the customer.
- **Prompt caching:** if a production prompt grows past 1,024 tokens (e.g. more policies), keep the static system prompt and policies *first* and the per-ticket data *last*, so the prefix gets billed at 0.3 instead of 3.
