# PersonaHire AI - Cost Breakdown Analysis

This document provides a detailed cost breakdown and budget estimation for deploying and running the **PersonaHire AI** candidate representative and interview scheduling agent.

---

## 1. Unit Pricing Overview

| Service Component | Provider | Model / Tier | Pricing Unit | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Embeddings** | OpenAI | `text-embedding-3-small` | $0.000020 per 1,000 tokens | 1536-dimensional semantic representation. |
| **LLM Synthesis** | OpenAI | `gpt-4o-mini` | Input: $0.000150 per 1k tokens<br>Output: $0.000600 per 1k tokens | Optimized for low latency and high cost-efficiency. |
| **Voice Platform** | Vapi | WebRTC Voice Pipeline | $0.050 per minute | Includes Vapi connection, concurrent routing, and tools. |
| **Speech-to-Text (STT)**| Deepgram | `nova-2` | $0.0043 per minute | High accuracy, sub-100ms real-time audio transcription. |
| **Text-to-Speech (TTS)**| ElevenLabs| `eleven_flash_v2` | $0.015 per 1,000 characters | Custom voice generation (approx. $0.08 per call minute). |
| **Vector Database** | ChromaDB | Local Persistence | **Free (Open Source)** | Runs locally in memory or is persistent inside disk file. |
| **Web UI Hosting** | Streamlit | Streamlit Cloud | **Free** | Public deployment. |
| **Backend API Hosting**| Render | Web Service Free Tier | **Free** (or $7.00/mo for Starter) | Free tier has cold start; Starter is recommended for voice. |

---

## 2. Ingestion Phase Cost (One-time Setup)

Calculated based on indexing:
- 1 Resume PDF (approx. 3 pages, ~2,000 words, ~3,000 tokens)
- 15 GitHub repositories overview and README files (approx. 20,000 tokens)
- 100 recent commit entries (approx. 10,000 tokens)

* **Total tokens ingested:** ~33,000 tokens.
* **Embedding Ingestion Cost:** 33,000 tokens × $0.000020/1k = **$0.00066 (Under 0.1 cents)**.

---

## 3. Interaction Phase Cost (Ongoing Usage)

### Scenario A: Web Chat Session (10 Multi-turn Questions)
An average web session consists of 10 turns.
- Average input per turn (query + context + memory): ~1,500 tokens.
- Average output per turn: ~200 tokens.

| Action | Token Vol | Cost Calculation | Session Cost |
| :--- | :--- | :--- | :--- |
| **Query Embedding** | 10 × 100 tokens | 1k tokens × $0.00002/1k | $0.00002 |
| **LLM Input (GPT-4o-mini)**| 10 × 1,500 tokens | 15k tokens × $0.00015/1k | $0.00225 |
| **LLM Output (GPT-4o-mini)**| 10 × 200 tokens | 2k tokens × $0.00060/1k | $0.00120 |
| **Total Chat Session Cost**| | | **$0.00347 (approx. 0.35 cents)**|

*For $1.00, the system can handle **over 280 complete 10-turn web chat interviews**.*

---

### Scenario B: Voice Screening Call (5 Minutes Duration)
Voice interaction adds Vapi, Deepgram STT, and ElevenLabs TTS costs on top of the RAG LLM queries.
Assuming 5 minutes of conversational duration (approx. 12 turns, ~150 words/min output, ~3,500 characters total).

| Component | Cost Calculation | Call Cost |
| :--- | :--- | :--- |
| **Vapi Platform Fee** | 5 mins × $0.05 / min | $0.25 |
| **Deepgram STT Fee** | 5 mins × $0.0043 / min | $0.02 |
| **ElevenLabs TTS Fee** | 3,500 characters × $0.015 / 1k chars | $0.05 |
| **LLM + Embeddings (12 turns)**| Input/Output token calculations | $0.01 |
| **Total Voice Call Cost** | | **$0.33 (approx. 33 cents)** |

*For $10.00, you can conduct **approx. 30 comprehensive 5-minute voice screening calls**.*

---

## 4. Production Monthly Hosting Projection (Starter Scale)

For a production recruitment deployment hosting 200 candidate screenings per month (150 web chat sessions + 50 voice screening calls):

1. **RAG & LLM (OpenAI API):** 150 × $0.0035 + 50 × $0.01 = **$1.03**
2. **Voice Pipelines (Vapi + TTS + STT):** 50 calls × $0.33 = **$16.50**
3. **Web UI Hosting (Streamlit Cloud):** Free = **$0.00**
4. **Backend API Hosting (Render Starter Instance):** **$7.00** (highly recommended to bypass the 50-second sleep cold-start on Render's free tier, ensuring low-latency voice pickup).

* **Total Estimated Monthly Budget:** **$24.53**
