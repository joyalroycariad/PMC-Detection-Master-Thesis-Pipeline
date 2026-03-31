# PMC-Detection-Master-Thesis-Pipeline

# 🚗 AI‑Powered PMC Detection Pipeline
**Master Thesis – CARIAD / Volkswagen Group**

This repository contains the full implementation of an **AI-driven PMC (Problem Management Candidate) Detection Pipeline**, built for analyzing IM/incident tickets and automatically identifying recurring problems across Volkswagen Group systems.

It is the end‑to‑end solution developed as part of my **Master Thesis**, combining  
data cleaning → error message extraction → embeddings → clustering → PMC creation → JSON payload building → GenAI‑based PMC summarisation.

---

## 📌 Key Features

- **Selective translation** (optional)
- **Data cleaning & preprocessing**
- **Error message extraction** (A#3 extraction + title fallback)
- **SBERT Embedding generation**  
  (`all‑MiniLM‑L6‑v2`)
- **DBSCAN clustering** with cosine similarity
- **Automatic PMC Candidate creation**
- **JSON payload generation per PMC cluster**
- **GenAI‑powered PMC summarisation** (optional)
- **Logging & metrics at each stage**
- **Modular + production‑ready folder structure**

---

## 📁 Project Structure

.
├── config/
│   └── config.yaml                 # Safe configuration (no API keys)
│
├── prompts/
│   └── pmc_prompt.txt              # Prompt template for summarisation
│
├── src/
│   ├── clustering/
│   │    └── clusterer.py           # DBSCAN clustering
│   │
│   ├── embeddings/
│   │    └── text_embeddings.py     # SBERT embeddings
│   │
│   ├── pmc/
│   │    ├── pmc_creation.py        # PMC candidate logic
│   │    ├── pmc_payload_builder.py # PMC JSON payload generator
│   │    └── pmc_summarizer.py      # LLM-driven summarisation
│   │
│   ├── preprocessing/
│   │    ├── cleaning.py            # Column drop, NA handling, date unification
│   │    ├── error_extraction.py    # A#3 extraction + fallback error logic
│   │    └── llm_description_cleaner.py
│   │
│   └── translation/
│        └── translator.py          # Optional translation module
│
├── main.py                         # Runs full pipeline end-to-end
├── pipeline.py                     # Pipeline controller / orchestrator
├── preprocess_pipeline.py          # Preprocessing-only pipeline
├── run_summarisation_only.py       # Summarise PMCs without recomputing embeddings
├── requirements.txt                # List of Python dependencies
└── README.md                       # Project documentation

---

## ⚙️ Configuration (config.yaml)

This file controls all pipeline behaviour.  
It defines:

- which dataset to load  
- whether translation is enabled  
- embedding model to use  
- clustering thresholds  
- PMC creation logic  
- JSON payload output paths  
- summarisation settings  

Below is an example configuration:

```yaml
data:
  input_path: "<your_dataset.xlsx>"

translation:
  enabled: false
  api_key: "......"
  region: "northeurope"
  endpoint: "https://api.cognitive.microsofttranslator.com"

embeddings:
  model: "all-MiniLM-L6-v2"
  batch_size: 32

clustering:
  eps: 0.14
  min_samples: 5

pmc:
  min_tickets: 5
  window_days: 21
  export_path: "pmc_clusters.xlsx"

payload:
  output_dir: "outputs/payloads"

summarisation:
  enabled: false
  prompt_file: "prompts/pmc_prompt.txt"
  client_id: "....."
  client_secret: "....."
  virtual_key: "...."
  model: "gpt-4.1-mini"
  limit: 2
  output_dir: "outputs/summaries"

▶️ Installation & Setup
1. Create a virtual environment

python -m venv venv
venv\Scripts\activate

2. Install required packages
pip install -r requirements.txt
``
3. Prepare your dataset
Place your Excel dataset in the project root (next to main.py),
and update this line in config.yaml:
data:
  input_path: "<your_dataset

4. Running the Full Pipeline
python main.py
This will:

  1. Load your dataset
  2. Clean titles/descriptions
  3. Extract raw + cleaned error messages
  4. enerate SBERT embeddings
  5. Cluster using DBSCAN
  6. Create PMC candidates
  7. Generate JSON payloads in  outputs/payloads/
  8. Export PMC supervisor Excel file: pmc_clusters.xlsx
  9. Pmc summarisation

