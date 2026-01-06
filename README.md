

RETAILMIND PROTOTYPE
LLM-Driven Conversation Diagnostics — Demo Pipeline & Dashboard Export

============================================================

This repository contains the RetailMind prototype: a reproducible, LLM-driven workflow for diagnosing and improving virtual assistant conversations.

The system analyzes conversational logs to:

1. score user satisfaction at the turn level
2. tag low-satisfaction turns with an issue taxonomy
3. cluster failures into interpretable topics
4. generate prompt repair and guardrail suggestions
5. export lightweight dashboard data for a frontend



---

## CORE DESIGN PRINCIPLES

* Reproducible end-to-end pipeline (single runner in src/)
* No large artifacts committed (parquet, caches, dashboard data)
* File-based LLM caching to avoid repeated cost
* Demo-friendly: runs on a small English subset in minutes
* Clear separation:

  * notebooks = exploration & reference
  * src/ = production-style pipeline

---

## REPOSITORY STRUCTURE

.
├── notebooks/
│   ├── llm_satisfaction.ipynb      (satisfaction scoring reference)
│   ├── issue_tagging.ipynb         (issue taxonomy tagging)
│   ├── topic_clustering.ipynb      (failure clustering)
│   ├── prompt_repair.ipynb         (prompt repair generation)
│   ├── dashboard_export.ipynb      (dashboard JSON export)
│   ├── ingestion*.ipynb            (dataset ingestion)
│   └── satisfaction.ipynb
│
├── src/
│   ├── run_pipeline.py             (MAIN demo pipeline runner)
│   ├── pipeline/
│   │   ├── io.py                   (centralized paths)
│   │   ├── llm_cache.py            (file-based LLM cache)
│   │   ├── stage_llm_satisfaction.py
│   │   ├── stage_issue_tagging.py
│   │   ├── stage_topic_clustering.py
│   │   └── stage_prompt_repair.py
│   ├── ingestion/
│   │   ├── load_data.py
│   │   ├── load_ccpe.py
│   │   ├── load_mwoz.py
│   │   ├── load_redial.py
│   │   └── load_redial_action.py
│   └── tagging/
│
├── data/
│   ├── raw/            (dataset sources, NOT committed)
│   ├── processed/      (generated parquet + cache, NOT committed)
│   └── dashboard/      (dashboard JSON exports, NOT committed)
│
├── requirements.txt
└── README.txt

---

## WHAT IS NOT COMMITTED (BY DESIGN)

* data/raw/            original datasets
* data/processed/      parquet outputs, embeddings, LLM cache
* data/dashboard/      dashboard JSON exports
* zip files shared externally (e.g., dashboard_data_demo.zip)



---

## SETUP

1. Create virtual environment (Windows PowerShell)

From repository root:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## API KEY SETUP (REQUIRED FOR LLM STAGES)

Create a .env file in the repository root:

```
OPENAI_API_KEY=your_key_here
```

The pipeline reads the key from an environment variable.

You can override the variable name (useful for temporary/demo keys):

```
python -m src.run_pipeline --api-key-env OPENAI_API_KEY1
```

---

## DEMO PIPELINE (RECOMMENDED ENTRY POINT)

The demo pipeline runs the full workflow on a small English subset.

Run full demo pipeline on 50 conversations:

```
python -m src.run_pipeline --demo-max-convs 50
```

This will:

* sample 50 English conversations
* run all pipeline stages
* cache all LLM calls
* write outputs locally to data/processed/

Re-run everything (ignore cached outputs):

```
python -m src.run_pipeline --demo-max-convs 50 --no-skip
```

Run only specific stages (example: clustering + repair only):

```
python -m src.run_pipeline --from-stage 3 --to-stage 4 --demo-max-convs 50
```

Pipeline stages:

1. LLM satisfaction scoring
2. Issue tagging
3. Topic clustering
4. Prompt repair generation

---

## LOCAL OUTPUTS PRODUCED

After a demo run (locally only):

* uss_english_turns_satisfaction_*.parquet
* uss_english_issue_tagged_*.parquet
* uss_english_topics_rows_*.parquet
* uss_english_topics_summary_*.parquet
* uss_english_turns_topics_*.parquet
* uss_english_prompt_repairs_*.parquet
* data/processed/llm_cache/

None of these files are committed.

---

## DASHBOARD EXPORT

Run:

```
notebooks/dashboard_export.ipynb
```

This produces JSON/JSONL files such as:

* dashboard_turns.jsonl
* dashboard_topics.json
* dashboard_repairs.json
* dashboard_sandbox_cases.json

These are typically zipped and shared with the frontend team.

---

