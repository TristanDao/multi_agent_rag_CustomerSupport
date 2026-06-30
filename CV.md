# DAO PHUOC THINH

(+84) 964 208 627  
dpthinhmep2023@gmail.com  
LinkedIn | GitHub  
Thu Duc City, HCM

## SUMMARY

AI Engineer specializing in LLM applications, RAG pipelines, multi-agent systems, and model fine-tuning. Experienced in building scalable, production-grade AI solutions with Python, FastAPI, LangGraph, vector databases, LLM guardrails, observability (Langfuse), and cloud deployment on AWS/GCP — with additional experience in data analytics and quantitative research.

---

## WORK EXPERIENCE

### AI ENGINEER TRAINEE
**VINSMART FUTURE — AI PRACTICAL TALENT DEVELOPMENT PROGRAM BY VINGROUP & VINUNIVERSITY**

*Apr 2026 – Present*

- Trained in practical AI engineering at VinUniversity, covering LLM applications, RAG, multi-agent systems, evaluation, observability, cloud deployment, and cost/latency optimization.
- Interned at VinSmart Future on Guardrail/model routing for LLM applications.
- Designed a multi-tier routing system that classifies query difficulty to optimize cost-quality tradeoffs across LLM deployments.
- Generated structured synthetic training data covering policy compliance edge cases including single-rule, multi-rule, borderline, and non-violation scenarios.
- Implemented LLM-as-a-Judge evaluation framework to automatically assess response quality and safety compliance at scale.

### AI QUANT RESEARCH INTERN
**FINPROS INVESTMENT COMPANY (FINANCE)**

*Oct 2025 – Mar 2026*

- Built ML-driven quantitative research pipelines for Vietnam derivatives markets using historical OHLCV and trading data.
- Engineered time-series feature pipelines including returns, volatility, momentum, volume patterns, and session-based indicators for predictive modeling.
- Trained and backtested statistical and machine learning models for short-term market movement forecasting.
- Integrated validated prediction and signal generation components into real-time trading workflows with preprocessing, monitoring, and regime-aware adaptation.

### DATA ANALYST
**GIA KHO GROUP COMPANY (ECOMMERCE, RETAIL)**

*Nov 2024 – Jul 2025*

- Built and maintained real-time ETL pipelines with Mage, Python, and SQL, integrating data from BigQuery, KiotViet, Google Ads, Facebook Ads, TikTok Ads, and Google Analytics.
- Developed interactive dashboards in Looker and Google Sheets.
- Optimized data workflows and marketing performance reporting.

---

## EDUCATION

### UNIVERSITY OF INFORMATION TECHNOLOGY – VNUHCM

Artificial Intelligence (Second Degree, Evening Program)

- GPA: 3.5/4.0
- Expected Sep 2026

### HCMC UNIVERSITY OF TECHNOLOGY AND EDUCATION

Thermal Engineering Technology

- Graduated Oct 2023

---

## TECHNICAL SKILLS

**Programming:** Python, C++, SQL, Bash

**AI/ML Frameworks:** PyTorch, Hugging Face, LangGraph, LlamaIndex

**LLM & NLP:** Embeddings, Prompt Engineering, RAG, Hybrid Retrieval, Reranking, Agentic Workflows, Fine-tuning, LLM Evaluation, Guardrails, PII Redaction, Model Routing

**Vector Databases & Data:** Chroma, Qdrant, Supabase, MongoDB, BigQuery, SQL Server

**LLM Serving & MLOps:** vLLM, FastAPI, Docker, Git, CI/CD, Terraform, Langfuse

**Cloud:** AWS, GCP

**Languages:** English (TOEIC 600, technical reading proficiency)

---

## PERSONAL PROJECTS

### MULTI-AGENT RAG E-COMMERCE SUPPORT ASSISTANT

**Goal:** Production-ready multi-agent RAG for wholesale customer support — grounds answers in product/order/policy data with guardrails and observability.

**Tech Stack:** Python, FastAPI, LangGraph, Alibaba DashScope (Qwen), Qdrant, PostgreSQL, Langfuse, Docker, Terraform, GCP, Streamlit.

- Architected a LangGraph `StateGraph` orchestrator wiring specialized agents (product, order, policy-RAG, sales, refund, response) via a declarative Agent Registry driving routing and graph nodes.
- Implemented hybrid retrieval (BM25 + dense embeddings → `gte-rerank`) with metadata filtering over policy documents.
- Built input/output guardrails: EN/VI prompt-injection detection, PII redaction, and unsupported-claim checks.
- Integrated Langfuse auto-tracing for LLM/tool/retrieval steps with cost & latency, and Postgres checkpointer for resumable, auditable conversations.
- Provisioned GCP infrastructure with Terraform: Cloud SQL (PostgreSQL 16) and Cloud Run for the FastAPI backend.

Source link

---

### VIETNAMESE READING COMPREHENSION — LLM FINE-TUNING

**Goal:** Fine-tune a <1B-param LLM for Vietnamese multiple-choice reading comprehension, improving accuracy from 0.54 → 0.9X.

**Tech Stack:** Python, PyTorch, Qwen2.5-0.5B, QLoRA (PEFT), vLLM, Hugging Face, GCP.

- Fine-tune Qwen2.5-0.5B with QLoRA on 1,500 Vietnamese news articles + MCQs, leveraging the train set's `explanation` field for chain-of-thought training to reason over distractor answers.
- Generate synthetic Q&A pairs (with explanations) via a strong LLM to augment underrepresented topics and balance the training distribution.
- Add passage retrieval for long articles to fit the small model's context window and reduce noise before answering.
- Build an eval harness (accuracy + per-question-type breakdown + LLM-as-a-Judge error analysis) on a held-out set.
- Target improving accuracy from 0.54 (prior baseline) to 0.9X, approaching the public benchmark SOTA of 0.9.

Source link

---

## CERTIFICATES

- IBM Data Analyst
- SQL Associate Certificate by DataCamp
- Extract, Transform and Load Data in Power BI by Microsoft
- Data Analyst (MindX Technology and Startup School)