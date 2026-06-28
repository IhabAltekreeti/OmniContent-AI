# 🎬 OmniContent AI

> Keyword in. Viral video out.

A fullstack AI system that takes a single keyword and turns it into a ready-to-post, vertical (1080x1920) short-form video — fully automated, from viral-potential scoring to final render. Built solo, on a zero-budget stack (Google Colab + free-tier APIs).

This is a **portfolio project**, not a production SaaS. It's built to demonstrate end-to-end AI/automation engineering — from ML model training to multi-agent orchestration to media rendering to a working web UI. Where it's a prototype rather than a production system, that's called out explicitly in [Known Limitations](#-known-limitations--engineering-tradeoffs) below.

---

## 📺 Demo

**Input keyword:** `"sad life"`
**Output:** a fully rendered vertical video with AI-written script, voiceover, and matched stock footage — no manual editing.

📹 **[Click to watch the demo video](assets/demo.mp4)**

*(GitHub doesn't autoplay video files in a README — click the link above to view/download it directly from the repo.)*

---

## 🏗️ Architecture

The system is a 4-layer pipeline. Each layer is a separate, independently testable module.

```
Keyword Input
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Layer 1 — Viral Score Predictor                │
│  Keras MLP trained on a Kaggle social-media     │
│  engagement dataset → predicts viral potential  │
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Layer 2 — Multi-Agent Script Generation        │
│  LangChain LCEL chain: Researcher → Scriptwriter│
│  → Brand Guard, with ChromaDB + sentence-       │
│  transformers RAG, served via OpenRouter        │
│  (Llama-3.2-3b-instruct, free tier)             │
└─────────────────────────────────────────────────┘
     │ approved script (JSON)
     ▼
┌─────────────────────────────────────────────────┐
│  Layer 3 — Media Rendering Pipeline             │
│  gTTS (voiceover) + Pexels API (stock footage)  │
│  + MoviePy/FFmpeg → final 1080x1920 MP4         │
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Layer 4 — Web UI                               │
│  Flask + Tailwind, served via pyngrok           │
│  Progress bar, viral-score gauge, script        │
│  preview, video player                          │
└─────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Engine | Python, Keras/TensorFlow, scikit-learn, Pandas, NumPy |
| AI Orchestration | LangChain (LCEL), OpenRouter (Llama-3.2-3b-instruct), ChromaDB, sentence-transformers |
| Media Pipeline | gTTS, Pexels API, MoviePy, FFmpeg |
| Web UI | Flask, Tailwind CSS, pyngrok |
| Environment | Google Colab (free tier, zero infrastructure cost) |

## 🚀 Running It

This project runs end-to-end inside **Google Colab** — no local setup required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IhabAltekreeti/OmniContent-AI/blob/main/OmniContent_AI.ipynb)

### 1. Get your API keys (all free tier)
| Key | Where to get it |
|---|---|
| `OPENROUTER_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) |
| `PEXELS_KEY` | [pexels.com/api](https://www.pexels.com/api/) |
| `NGROK_TOKEN` | [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken) |

### 2. Add them to Colab Secrets
In the Colab notebook, open the 🔑 **Secrets** tab in the left sidebar and add all three keys with **exactly these names**: `OPENROUTER_KEY`, `PEXELS_KEY`, `NGROK_TOKEN`. Make sure "Notebook access" is toggled on for each.

The notebook reads them via `google.colab.userdata` — your keys are never written into any cell or committed to this repo.

### 3. Run the cells top to bottom
Cell order matters — see the in-notebook section headers. The first run will also train the viral-score model (Layer 1); this takes a couple of minutes.

> **Dataset for training:** The notebook expects `Social_Media_Engagement_Dataset.csv` at Colab's `/content/` folder. This repo includes a copy of that dataset at `omnicontent/model/data/Social_Media_Engagement_Dataset.csv` — download it from here and upload it to `/content/` in your Colab session before running the training cell. (The trained model files — `viral_mlp.keras` and `scaler.pkl` — are already included in this repo, so retraining is only needed if you want to reproduce the training step yourself.)

> This project is Colab-only by design — it's built, tested, and meant to be run inside Google Colab, not on a local machine. The `.env.example` and `requirements.txt` in this repo document the dependencies for reference, but no local setup or support is provided.

## 📸 How It Works

1. Enter a keyword (e.g. `"sad life"`)
2. Layer 1 scores its viral potential
3. Layer 2's agent chain researches the topic, writes a script, and a "Brand Guard" agent reviews it before approval
4. Layer 3 generates voiceover + fetches matching stock footage + renders the final MP4
5. Layer 4's UI shows live progress and lets you preview/download the result

---

## ⚠️ Known Limitations & Engineering Tradeoffs

This section is here on purpose. A project that pretends everything is production-ready is less convincing than one that's explicit about where the corners were cut and *why*.

**Data leakage was caught and removed, not just avoided by luck.**
The original Kaggle dataset used to train the viral-score model includes columns like `likes_count` and `shares_count` — metrics that only exist *after* a post has already been published and gone viral. Training on those would mean predicting virality using virality. It's a textbook data leakage trap, and an easy one to miss under deadline pressure: it inflates a model's accuracy on paper while making it useless for the actual use case (scoring a keyword *before* anything is published). These columns were identified and excluded from the training features for that reason — the model only ever sees pre-publication signals.

**The dataset is a proxy, not real production data.**
The Kaggle dataset is a reasonable stand-in for learning the pipeline end-to-end, but it doesn't reflect the messiness of real-world content data (platform-specific algorithm shifts, audience drift over time, survivorship bias in what even got published). In a real product, this model would need to be retrained continuously on live, first-party engagement data — not a static public dataset — and validated against actual campaign outcomes, not just held-out test accuracy.

**`moviepy==1.0.3` is pinned to an old major version.**
The code uses `from moviepy.editor import ...`, which was removed in MoviePy 2.x. Upgrading the dependency would break the rendering pipeline as-is. This is a known tradeoff, not an oversight — a migration to 2.x's new import structure is a planned follow-up, not done here to avoid scope creep on a portfolio deadline.

**No containerization yet.**
There's no Dockerfile and no HuggingFace Spaces deployment live. The project currently runs from the Colab notebook. Containerizing it for one-command deployment is a natural next step.

**Free-tier constraints shape some choices.**
The LLM calls run on OpenRouter's free tier (Llama-3.2-3b-instruct), which is rate-limited and less capable than larger hosted models. This was a deliberate cost constraint, not a technical ceiling — swapping to a paid model is a one-line config change in `pipeline/agents.py`.

**A numpy binary-incompatibility warning may appear on first run.** Installing TensorFlow and Keras via pip inside the same Colab cell that upgrades them can occasionally leave a stale numpy binary loaded in memory, producing a `numpy.dtype size changed` error during model training. If this happens, restart the Colab runtime (Runtime → Restart runtime) and re-run the cells from the top — this is a one-time environment quirk, not a bug in the training code itself.

---

## 📂 Project Structure

```
OmniContent-AI/
├── omnicontent/             # Main package — run as `python -m omnicontent.app` from repo root
│   ├── config.py
│   ├── app.py               # Flask routes + Tailwind UI
│   ├── model/
│   │   ├── train.py         # Trains the viral-score MLP
│   │   ├── predictor.py     # Loads model + scaler for inference
│   │   ├── viral_mlp.keras
│   │   └── scaler.pkl
│   └── pipeline/
│       ├── agents.py        # LangChain multi-agent script generation + RAG
│       └── video.py         # gTTS + Pexels + MoviePy rendering
├── assets/
│   └── demo.mp4
├── .env.example
├── .gitignore
├── requirements.txt
├── LICENSE
├── README.md
└── OmniContent_AI.ipynb     # Original Colab notebook
```

> All file paths in the code (`.env`, model files, output videos) are relative to the **repo root**, and they include the `omnicontent/` prefix where the package's own files live — for example, `omnicontent/model/viral_mlp.keras`. This matches how the notebook runs in Colab, where the working directory is the project root.

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

Built by **Ihab Altekreeti** — 3rd-year Electrical-Electronics Engineering student, self-taught developer.

[GitHub](https://github.com/IhabAltekreeti)
