# RetailMind  
### LLM-Driven Conversation Diagnostics for Chatbot Teams

RetailMind is an AI diagnostic layer that analyzes chatbot conversations and explains **why they fail — and how to fix them**.

Instead of dashboards that only show metrics, RetailMind identifies root causes, clusters recurring issues, and generates actionable prompt and training improvements for developers.

---

## The Problem

Chatbot teams review thousands of logs manually to understand failures.  
This process is:

- Slow (1–2 days per 1,000 chats)
- Reactive
- Non-scalable
- Lacking structured root-cause insights

Dashboards show numbers — not reasons.

---

## The Solution

RetailMind transforms raw chat logs into structured diagnostic intelligence.

It:

- Predicts user satisfaction at turn level
- Tags low-satisfaction turns with failure categories
- Clusters recurring issues into interpretable topics
- Generates improved replies and prompt patches
- Ranks suggested fixes by predicted impact
- Exports insights for a developer-facing dashboard

---

## High-Level Workflow

Raw Chat Logs  
→ Ingestion & normalization  
→ Satisfaction scoring  
→ LLM issue tagging  
→ Embedding + topic clustering  
→ Prompt repair generation  
→ Candidate ranking  
→ Dashboard insights for developers  

---

## Core Capabilities

- Turn-level satisfaction prediction  
- Explainable failure taxonomy  
- Root-cause topic discovery  
- Automated prompt repair generation  
- Impact-based candidate ranking  
- Developer-oriented dashboard export  

---

## Tech Stack

- Python (Pandas, PyArrow)  
- Hugging Face Transformers  
- OpenAI GPT models  
- Sentence Transformers + BERTopic  
- Parquet-based artifact storage  

---

## Status

Prototype developed as part of the Reply Challenge @ Politecnico di Torino.  
Focused on B2B chatbot diagnostics and developer intelligence.
