# clinical-agent

An agentic clinical reasoning system. Given a free-text clinical question,
an LLM-based agent decides which tools to call, executes them, reasons over
their outputs, and produces a sourced answer with an inspectable trajectory.

**Companion to [`clinical-llm`](https://github.com/z-awan-lab/clinical-llm)**,
which baselined four models on ICU mortality prediction. Where `clinical-llm`
asks *how well can a model predict outcomes?*, this asks *how well can a
model reason over heterogeneous clinical knowledge sources?*

**Domain focus:** sepsis. Same patient population as `clinical-llm`'s ICU
mortality work.

**Design principle:** open-only, no external API calls at inference time —
deployable behind a hospital firewall.

> ⚠️ This is a portfolio engineering artefact. It is not for clinical use,
> decision-making, or patient-facing deployment.

---

## Current state

**Phase 1 of 6 complete.** Scaffolding + `pubmed_search` tool.

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | Scaffolding, CI, Docker, base classes, `pubmed_search` tool | ✅ |
| 2 | `guideline_retrieval` tool (BGE-large + Qdrant + sepsis corpus) | ⏳ |
| 3 | `clinical_calculator` tool (qSOFA, SOFA, NEWS2, …) | ⏳ |
| 4 | LangGraph orchestrator + MedGemma 1.5 27B-IT integration | ⏳ |
| 5 | Full evaluation suite + `results.md` | ⏳ |
| 6 | Streamlit two-pane demo + final polish | ⏳ |

See [`design.md`](./design.md) for the full specification.

---

## Architecture (target)

```
User question
  → LangGraph state machine
      ├─ plan      → agent decides next action
      ├─ tool_*    → guideline_retrieval | clinical_calculator | pubmed_search
      ├─ reflect   → review output, decide continue or finalise
      └─ answer    → response with citations (or INSUFFICIENT_EVIDENCE)
```

Three tools, deliberately small for v1:
- **`guideline_retrieval`** — RAG over NICE NG51, CDC, WHO, NIH sepsis content
  with evidence-tier metadata
- **`clinical_calculator`** — deterministic scoring functions (qSOFA, SOFA,
  NEWS2 plus distractors to evaluate tool-selection honesty)
- **`pubmed_search`** — NCBI E-utilities for primary literature

---

## Quickstart

```bash
git clone git@github.com:z-awan-lab/clinical-agent.git
cd clinical-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the test suite
pytest -v

# Try the PubMed tool against the real API
export NCBI_API_KEY="your-key-here"  # optional, gives 10 req/sec vs 3 req/sec
python scripts/try_pubmed.py "sepsis[MeSH] AND mortality" --max-results 3
```

### Docker

```bash
docker compose build
docker compose run --rm app pytest -v
```

---

## Repository layout

```
src/clinical_agent/
├── tools/          # BaseTool + pubmed_search (Phase 1) + ... (Phases 2-3)
├── embeddings/     # BaseEmbedder (Phase 2)
├── vectorstore/    # BaseVectorStore (Phase 2)
├── generation/     # BaseGenerator (Phase 4)
└── utils/          # logging helpers

tests/              # unit + offline-mocked integration; live tests gated by env var
scripts/            # try_pubmed.py and (later) build_index.py, run_eval.py
configs/            # default.yaml; layered configs added as components land
```

## Design decisions worth flagging

- **Open-only constraint.** No frontier API calls. Comparison context in
  `results.md` comes from published benchmark numbers, not paid runs.
- **Tools return a `ToolResult` envelope**, not raw values or exceptions.
  Expected failures (no results, HTTP errors) populate `error`; only
  programmer-fault errors bubble up. Makes the orchestrator's life simpler.
- **Pluggable backends** via abstract base classes — concrete implementations
  register themselves and are selected by config string. Same pattern as
  `clinical-llm`.
- **Tests run offline.** PubMed XML fixtures live in `tests/fixtures/`. A
  single live integration test is gated by `CLINICAL_AGENT_INTEGRATION_TESTS=1`.

---

## Evaluation (planned, lands in Phase 5)

All metrics reported with bootstrap 95% CIs, same convention as `clinical-llm`.

- **End-task accuracy** — MedQA subset, PubMedQA, hand-curated sepsis set
- **Trajectory quality** — tool-selection accuracy, step count, redundancy,
  failed-tool-call rate
- **Citation faithfulness** — RAGAS adapted for trajectories
- **Safety / refusal** — refusal rate on out-of-corpus questions; false-refusal
  rate on in-scope questions

Models compared: **MedGemma 1.5 27B-IT** (default) and **Qwen 2.5 72B-Instruct**
(general baseline). External anchor: published frontier-model numbers from
MedAgentBench (Jiang et al., 2025).

---

## Licensing notes

Code is Apache 2.0. The repository does **not** redistribute corpus content.
Ingestion scripts download from source; manifests record URLs and content
hashes for reproducibility.

- CDC, NIH MedlinePlus: US government public domain
- WHO: CC BY-NC-SA IGO
- NICE: UK Open Content Licence (research use; attribution required)
- PubMed: free for research via E-utilities API
