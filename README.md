# retailmind-prototype
## Demo Repro (English subset)

### 1) Setup
Create a `.env` file in the repo root (not committed):

```bash
OPENAI_API_KEY1=temp_key
# OPENAI_API_KEY=reply_key_if_working

# RetailMind Prototype — Demo Pipeline + Dashboard Export

This repository contains the RetailMind prototype codebase and notebooks for building a developer-oriented diagnostics workflow:
1) compute user satisfaction signals,
2) tag low-satisfaction turns with issue taxonomy,
3) cluster failures into topics,
4) generate prompt repair packages,
5) export lightweight dashboard JSON files for the frontend.


The project is organized so that:
- **Notebooks** are reference implementations / exploration,
- **`src/`** contains a reproducible **pipeline runner** that can regenerate outputs locally (without committing large parquet files),
- **Dashboard exports** are produced as JSON/JSONL and shared via zip for the frontend.

---

## Repository Structure

.
├── notebooks/
│ ├── llm_satisfaction.ipynb
│ ├── issue_tagging.ipynb
│ ├── topic_clustering.ipynb
│ ├── prompt_repair.ipynb
│ ├── dashboard_export.ipynb
│ ├── ingestion*.ipynb
│ └── satisfaction.ipynb
│
├── src/
│ ├── run_pipeline.py
│ ├── pipeline/
│ │ ├── io.py
│ │ ├── llm_cache.py
│ │ ├── stage_llm_satisfaction.py
│ │ ├── stage_issue_tagging.py
│ │ ├── stage_topic_clustering.py
│ │ └── stage_prompt_repair.py
│ ├── ingestion/
│ │ ├── load_data.py
│ │ ├── load_ccpe.py
│ │ ├── load_mwoz.py
│ │ ├── load_redial.py
│ │ └── load_redial_action.py
│ └── tagging/
│ └── (tagging utilities)
│
├── data/
│ ├── raw/ # dataset sources (not committed)
│ ├── processed/ # generated parquet outputs (not committed)
│ └── dashboard/ # generated dashboard JSON exports (not committed)
│
├── requirements.txt
└── README.md


Notes:
- `data/processed/` contains generated `.parquet` outputs and an LLM cache directory. These are **not committed**.
- `data/dashboard/` contains generated JSON exports for the frontend. These are **not committed**.
- Zip files like `dashboard_data_demo.zip` are used to share exports externally (Telegram/Drive) and are **not committed**.

---

## Setup

### 1) Create environment (Windows PowerShell)
From repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
