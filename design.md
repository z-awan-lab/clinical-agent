# clinical-agent — design

## Purpose

This document captures the design decisions for `clinical-agent`, a follow-on
to `clinical-llm`. It is the spec we work to. Decisions here can be revised,
but only deliberately.

## What this project is

A clinically grounded agentic reasoning system focused on sepsis. Given a
free-text clinical question, an LLM-based agent decides which tools to call,
executes them, reasons over their outputs, and produces a sourced answer with
an inspectable trajectory.

Use case framing: clinical question answering with tool use. The system is
benchmarked on this. A vignette-style demo runs on the same engine for
qualitative demonstration.

## Design principles

1. **Open-only, deployable behind a hospital firewall.** No external API calls
   at inference time. The whole system runs on a single GPU host a hospital
   could plausibly operate. This is a relevant production constraint for
   regulated clinical environments, not an arbitrary limitation.
2. **Honest evaluation over impressive numbers.** Bootstrap 95% CIs on every
   reported metric — same convention as `clinical-llm`. Report what the
   evaluation actually says, including where the system underperforms.
3. **Pluggable components.** Reasoning model, embedder, vector store, and
   tools are swappable via config. Same `Base*` abstract-class pattern as
   `clinical-llm`.
4. **Determinism where appropriate.** Calculator outputs are deterministic
   Python. The agent invokes them; it never rederives them in the LLM. This
   distinction is a deliberate clinical-safety design choice.
5. **Tight v1 scope.** Three tools. One clinical domain (sepsis). One
   reasoning model as default, one as comparison. v2 features are listed as
   future work, not promised.

## Architecture

```
User question
  → LangGraph state machine
      → plan (agent decides next action)
      → one of: tool_guideline_retrieval | tool_clinical_calculator | tool_pubmed_search
      → reflect (review output, decide continue or finalise)
      → answer | refuse
```

Agent state:
- Original question
- Reasoning scratchpad
- Tool call history (inputs and outputs)
- Retrieved evidence with provenance and evidence tier
- Termination signal

Termination conditions:
- Agent emits a final answer with citations
- Step budget exceeded (default: 8)
- Repeated identical tool call detected (loop guard)
- `INSUFFICIENT_EVIDENCE` refusal triggered

## Tools (v1)

### `guideline_retrieval`
Returns top-k chunks from the sepsis corpus with source, evidence tier, and
chunk ID.
- Embedder: BGE-large-en-v1.5 (default), MedCPT and Qwen3-Embedding-8B as
  comparison runs in evaluation
- Vector store: Qdrant in Docker, with payload filtering by `source_type`
  and `evidence_tier`
- Chunking: 512 tokens / 64 overlap, recursive on section headers for
  guidelines; abstract-as-unit for PubMed

### `clinical_calculator`
Deterministic scoring functions, pure Python, exhaustively unit-tested.
- Sepsis-relevant: qSOFA, SOFA, NEWS2, Sepsis-3 criteria check
- Distractors (kept in tool set to evaluate tool-selection honesty):
  CHA2DS2-VASc, Wells DVT

### `pubmed_search`
NCBI E-utilities (eSearch + eFetch). Returns ranked abstracts with PMIDs,
MeSH terms, publication dates, and excerpt. Requires NCBI API key.

### Explicitly out of scope for v1
- Drug interaction checking
- Lab reference range lookup
- EHR / FHIR integration (full MedAgentBench environment is too heavy)
- Multi-agent architectures

## Corpus

Scope: sepsis. Specific sources to ingest:

| Source | Licence | Treatment |
|---|---|---|
| CDC sepsis pages | Public domain (US gov) | Direct scrape, full ingest |
| NIH MedlinePlus (sepsis) | Public domain (US gov) | Direct scrape |
| WHO sepsis guidance | CC BY-NC-SA IGO | Ingest with attribution |
| NICE NG51 + related QS | NICE UK Open Content Licence | Ingest under UK research use; cite explicitly; do not redistribute |
| Surviving Sepsis Campaign 2021 | Check terms before commit | Defer if licensing unclear |
| PubMed sepsis abstracts | Public via E-utilities | ~2,000 abstracts, MeSH "Sepsis", last 10 years |

**The repository does NOT commit raw corpus content.** Ingestion is
reproducible via `scripts/ingest_*.py` plus a `corpus_manifest.json`
recording source URLs and content hashes. README documents licensing.

Every chunk carries:
- `source_type` ∈ {`guideline`, `consumer_health`, `research_abstract`}
- `evidence_tier` (manually assigned to source, propagated to chunks)
- `source_url`, `retrieved_at`, `chunk_id`

## Models

**Default reasoning agent: MedGemma 1.5 27B-IT, 4-bit quantised.** Strongest
open medical model in 2026; Gemma family for continuity with Project 1.

**Comparison reasoning agent: Qwen 2.5 72B-Instruct, 4-bit quantised.** Strong
general-purpose tool use. Answers the implicit question: does medical
pretraining help when medical knowledge is being tool-injected anyway?

Both fit on a single H100 with 4-bit quantisation.

**No frontier API models are used.** Comparison context in `results.md` comes
from published benchmark numbers (e.g. Jiang et al. 2025 MedAgentBench
evaluations of GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Pro, etc.) cited as
external anchors.

## Orchestration

LangGraph. State-machine model with explicit nodes:
- `plan` — agent decides next action
- `tool_guideline_retrieval` | `tool_clinical_calculator` | `tool_pubmed_search`
- `reflect` — review tool output, decide continue or finalise
- `answer` — final response with citations
- `refuse` — structured `INSUFFICIENT_EVIDENCE` output

State checkpointing at every node transition for replay and trajectory eval.

## Evaluation

All metrics reported with bootstrap 95% CIs.

### End-task accuracy
- **MedQA**: 200-question subset, USMLE-style. Headline accuracy number.
- **PubMedQA**: yes/no/maybe accuracy on test subset.
- **Hand-curated sepsis set**: 50 questions across guideline-grounded,
  calculator-required, literature-retrieval-required, and out-of-scope
  categories. Constructed against NICE NG51 + Surviving Sepsis as reference.

### Trajectory quality
- Tool-selection accuracy (correct tool given question category)
- Step count distribution
- Redundancy rate (repeated tool calls with similar args)
- Failed tool call rate (parse errors, malformed args)

### Citation faithfulness
- RAGAS faithfulness adapted for agent trajectories
- Judge: Qwen 2.5 72B running locally
- Judge validated against manual annotation on ~20 examples; calibration
  reported in `results.md`

### Safety / refusal
- ~20 deliberately out-of-corpus or out-of-scope questions
- Refusal rate via structured `INSUFFICIENT_EVIDENCE` output
- False-refusal rate on in-scope questions (also measured)

### Comparison
- Full eval suite on both MedGemma 27B and Qwen 72B
- Published frontier-model numbers from MedAgentBench cited as external anchor

## Demo

Streamlit, two-pane:
- **Left**: input box (free-text question or vignette), submit button,
  vignette presets for one-click demo
- **Right**: live-updating trajectory — each tool call with arguments and
  collapsed output, agent reasoning between calls, final answer with
  citations linking back to specific tool outputs. Tool types colour-coded.
  Timing per step shown.

The trajectory pane is the part interviewers will look at first. It is part
of the deliverable, not an afterthought.

## Repository structure

```
clinical-agent/
├── README.md
├── design.md                       # this file
├── results.md
├── Dockerfile
├── docker-compose.yml              # app + Qdrant
├── pyproject.toml
├── .github/workflows/ci.yml
├── configs/                        # YAML configs
├── src/clinical_agent/
│   ├── ingestion/                  # source-specific loaders + chunkers
│   ├── embeddings/                 # BaseEmbedder + BGE / MedCPT / Qwen3 impls
│   ├── vectorstore/                # BaseVectorStore + Qdrant adapter
│   ├── retrieval/                  # retriever, optional reranker
│   ├── generation/                 # BaseGenerator + MedGemma / Qwen wrappers
│   ├── tools/
│   │   ├── guideline_retrieval.py
│   │   ├── clinical_calculator.py
│   │   └── pubmed_search.py
│   ├── orchestrator/               # LangGraph state machine
│   ├── eval/                       # MedQA, PubMedQA, trajectory metrics, RAGAS
│   └── app/                        # Streamlit two-pane demo
├── scripts/                        # ingest_*.py, build_index.py, run_eval.py
├── tests/
├── data/                           # gitignored
└── notebooks/                      # exploratory only
```

## Milestone plan (6–8 weeks, 5–15 hrs/week)

| Phase | Work | Estimate |
|---|---|---|
| 1 | Scaffolding + `pubmed_search` tool | ~10 hrs |
| 2 | `guideline_retrieval` tool (full RAG-as-tool, Qdrant, ingestion) | ~14 hrs |
| 3 | `clinical_calculator` tool | ~6 hrs |
| 4 | LangGraph orchestrator + MedGemma 27B integration | ~16 hrs |
| 5 | Evaluation suite + `results.md` (both models, all four eval tracks) | ~18 hrs |
| 6 | Streamlit demo + README polish | ~12 hrs |
| 7 (opt) | Drug interaction tool or DSPy prompt-optimization experiment | — |

Floor: 76 hours. Ceiling (with stretch): 95 hours.

Each phase produces commitable units. Sessions end with a push.

## Disclaimers in the README

> This system is a portfolio engineering artefact. It is not for clinical
> use, decision-making, or patient-facing deployment. It demonstrates
> tool-using agent design, evaluation methodology, and clinical-safety
> design considerations.

## Open questions deferred to implementation

- Whether to add a reranker (e.g. BGE-reranker-v2) in the retrieval tool.
  Decide after first retrieval-only eval numbers.
- LangGraph human-in-the-loop primitives for the refusal path: probably
  not for v1.
- Whether the trajectory pane exposes the raw scratchpad to the demo
  viewer (likely yes, with UX consideration).
- Whether to ship MedCPT and Qwen3-Embedding-8B as live alternates or
  only run them in evaluation. Default: eval-only.

## Future work (explicit non-promises)

- DSPy compilation of the agent prompt for measured improvement
- Drug interaction tool via OpenFDA or RxNav
- Lab reference range tool
- Multi-domain corpus (heart failure, AF) once sepsis baseline is locked
- Multi-agent variant with specialised sub-agents
