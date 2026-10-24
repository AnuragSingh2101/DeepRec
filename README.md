# DeepRec — A Two-Stage Deep Learning Video Recommendation System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Research Foundation:** Inspired by the landmark industrial recommendation paper:  
> *“Deep Neural Networks for YouTube Recommendations”*  
> **Authors:** Paul Covington, Jay Adams, Emre Sargin (Google, ACM RecSys 2016).

---

## 1. Project Overview & Problem Statement

Recommendation systems operate under extreme scale constraints. Delivering relevant, low-latency recommendations across millions of items cannot be accomplished with a single dense neural network. 

**DeepRec** implements Google's classic **Two-Stage Recommendation Architecture**:
1. **Stage 1 — Candidate Generation (Nomination):** Filters the catalogue of thousands/millions of items down to hundreds of high-recall candidates in sub-millisecond time using sequence-to-item embeddings and masked average pooling.
2. **Stage 2 — Deep Ranking (Scoring):** Employs a fine-grained deep neural network with continuous feature normalizations, power transforms ($x, x^2, \sqrt{x}$), and collaborative/genre signals to score and rerank candidates into the final top-N personalized feed.

DeepRec is built using **Python, PyTorch, FastAPI, and Streamlit**, complete with strict chronological sequence generation, leakage prevention, benchmark baselines, offline evaluation metrics (Recall@K, NDCG@K, Precision@K, MAP@K, Coverage, Latency), and interactive inspection.

---

## 2. System Architecture

```mermaid
graph TD
    A[Full Catalogue: 1,682 Items] --> B[Stage 1: Deep Candidate Generation]
    H[Chronological Watch History w_1 ... w_T] --> B
    C[User Demographics & Context: Age, Gender, Occupation] --> B
    
    B -->|Top-100 High Recall Candidates| D[Stage 2: Deep Neural Ranker]
    
    subgraph Feature Engineering (Train-Fitted)
        F1[Candidate Logit Score]
        F2[Item Train Popularity & Bayesian Mean Rating]
        F3[User History Stats & Interaction Recency]
        F4[User-Item Genre Cosine Affinity]
        F5[Power Transforms: x, x^2, sqrt x]
    end
    F1 --> D
    F2 --> D
    F3 --> D
    F4 --> D
    F5 --> D
    
    D -->|Calibrated Engagement Probability| E[Deduplication & Seen-Item Filter]
    E -->|Final Top-10 Personalized Feed| UI[Interactive Dashboard & REST API]
```

---

## 3. Two-Stage Pipeline Details

### Stage 1: Candidate Generation
- **Representation:** Sequence of item IDs embedded into $\mathbb{R}^d$, followed by masked average pooling that excludes padding token `0`.
- **Demographic Context:** Normalized continuous age projection, gender embedding, and occupation embedding concatenated with history representation.
- **Hidden Layers:** Configurable multi-layer perceptron (`[128, 64]` with LayerNorm, ReLU, and Dropout).
- **Output Layer:** Candidate item output embeddings $V$ and biases $b$.
- **Training Objective:** Extreme multiclass classification trained via **Sampled Softmax Cross-Entropy** with uniform negative sampling.
- **Inference:** Fast batched matrix dot product: $\mathbf{s} = u \cdot V^\top + b$, returning top-100 candidates in $<1$ ms.

### Stage 2: Deep Ranking
- **Inputs:** Candidate generation logit, item training popularity, item rating statistics, user historical mean rating, user interaction count, interaction recency (elapsed days), movie release year, and genre affinity.
- **Continuous Transforms:** Features are standardized using training data statistics and augmented with powers: $x, x^2, \sqrt{x}$ as specified in Section 4.1 of the paper.
- **Categorical Embeddings:** User ID and Movie ID dense embeddings.
- **Architecture:** Dense MLP (`[128, 64, 32]` with LayerNorm, ReLU, and Dropout) with a single sigmoid-activated output head.
- **Training Objective:** Binary Cross-Entropy with Logits on high-engagement interaction proxy (rating $\ge 4.0$).

---

## 4. Empirical Benchmark Results

Evaluated on the held-out temporal test set (final chronological interaction per user across 943 users):

| Experiment | Model Architecture | Recall@10 | Recall@50 | Precision@10 | NDCG@10 | MAP@10 | Catalog Coverage | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | Popularity Baseline | 0.0267 | 0.1100 | 0.0027 | 0.0117 | 0.0069 | 3.80% | 0.01 ms |
| **Exp B** | Item-Item Collaborative Filtering | 0.0533 | 0.2967 | 0.0053 | 0.0268 | 0.0182 | 62.96% | 0.06 ms |
| **Exp C** | Shallow Candidate Net [64] | 0.0433 | 0.1167 | 0.0043 | 0.0173 | 0.0089 | 15.22% | 0.77 ms |
| **Exp D** | **Deep Candidate Net [128, 64]** | **0.0867** | **0.2667** | **0.0087** | **0.0387** | **0.0232** | **49.23%** | **0.88 ms** |
| **Exp E** | Candidate Net + Context | **0.0867** | **0.2667** | **0.0087** | **0.0387** | **0.0232** | **49.23%** | **0.81 ms** |
| **Exp F** | Two-Stage (Candidate + Ranker, Pool=100) | 0.0367 | 0.2167 | 0.0037 | 0.0193 | 0.0137 | 49.35% | 2.13 ms |
| **Exp G1** | Two-Stage (Candidate Pool = 50) | 0.0533 | 0.2667 | 0.0053 | 0.0276 | 0.0194 | 49.23% | 1.94 ms |
| **Exp G2** | Two-Stage (Candidate Pool = 200) | 0.0300 | 0.1500 | 0.0030 | 0.0169 | 0.0127 | 47.98% | 2.75 ms |

> [!NOTE]  
> Deep Candidate Generation achieves **3.25x the Recall@10 of the Popularity Baseline** while discovering **13x more catalogue items** (49.2% vs 3.8% coverage) with a mean latency of only **0.88 ms**.

---

## 5. Repository Structure

```
DeepRec/
├── RDS_ProjectPaper.pdf            # Research paper reference
├── README.md                       # Comprehensive documentation
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── .gitattributes                 # Line-ending and binary tracking
├── configs/
│   └── model.yaml                  # Unified pipeline & hyperparameter configuration
├── data/
│   ├── raw/ml-100k/                # Extracted MovieLens dataset
│   └── processed/                  # Temporal train/val/test splits, mappings, metadata
├── artifacts/
│   ├── candidate/                  # Candidate model checkpoint & metadata
│   ├── ranking/                    # Ranking model checkpoint & feature extractor
│   └── evaluation/                 # Metrics JSON, benchmark CSVs, comparison plots
├── docs/
│   ├── paper_implementation_mapping.md # Section-by-section comparison with paper
│   ├── architecture.md             # Detailed network diagrams and math formulas
│   ├── data_dictionary.md          # Complete data and feature dictionary
│   ├── experiments.md              # Empirical results, ablations, latency analysis
│   └── limitations.md              # Dataset differences, assumptions, boundary conditions
├── src/
│   └── deeprec/
│       ├── __init__.py
│       ├── config.py               # YAML configuration and device loader
│       ├── data/
│       │   ├── download.py         # Dataset download & verification
│       │   ├── inspect.py          # Data inspection & shapes reporting
│       │   ├── preprocess.py       # Deterministic temporal split & ID mapping
│       │   ├── sequences.py        # Causal rolling history generator (no leakage)
│       │   ├── dataset.py          # PyTorch Datasets & DataLoaders
│       │   └── ranking_features.py # Feature extraction, normalization & powers
│       ├── models/
│       │   ├── candidate_generation.py # Deep Candidate Generator
│       │   ├── ranking.py          # Deep Ranking Network
│       │   └── baseline.py         # Popularity and Item-Item CF Baselines
│       ├── training/
│       │   ├── losses.py           # Sampled Softmax & BPR Loss
│       │   ├── train_candidates.py # Stage 1 training & checkpointing
│       │   └── train_ranker.py     # Stage 2 training & checkpointing
│       ├── retrieval/
│       │   └── retrieve.py         # Vectorized candidate retriever & cold start
│       ├── evaluation/
│       │   ├── metrics.py          # Precision, Recall, NDCG, MAP, Coverage
│       │   ├── evaluate.py         # Stage 1 and Two-Stage test evaluation
│       │   └── experiments.py      # Experiments A through G benchmark runner
│       ├── service/
│       │   └── recommender.py      # End-to-end two-stage service orchestrator
│       └── api/
│           └── main.py             # FastAPI REST endpoints
├── dashboard/
│   └── app.py                      # Interactive Streamlit Web Dashboard
└── tests/
    ├── test_data.py                # Preprocessing, filtering, and split tests
    ├── test_sequences.py           # Padding, masking, and leakage tests
    ├── test_candidate_model.py     # Shapes, forward pass, loss, checkpoint tests
    ├── test_ranking_model.py       # Ranking features, powers, and loss tests
    ├── test_metrics.py             # Known-answer metric tests
    ├── test_recommender.py         # Filtering, cold start, and schema tests
    └── test_api.py                 # FastAPI endpoints tests
```

---

## 6. Installation & Execution Guide (Windows PowerShell)

### Step 1: Environment Setup
```powershell
# Clone or navigate to repository root
cd C:\Users\hp\OneDrive\Desktop\DeepRec

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Verify Python and dependencies
python -m pip install -r requirements.txt
```

### Step 2: Data Pipeline
```powershell
# Download MovieLens 100K (verifies existing files before downloading)
python -m src.deeprec.data.download

# Inspect dataset shapes, missing values, and rating distributions
python -m src.deeprec.data.inspect

# Preprocess with zero-padding token and strict temporal split
python -m src.deeprec.data.preprocess
```

### Step 3: Model Training
```powershell
# Train Stage 1: Deep Candidate Generation Model
python -m src.deeprec.training.train_candidates --epochs 10

# Train Stage 2: Deep Ranking Model
python -m src.deeprec.training.train_ranker --epochs 10
```

### Step 4: Model Evaluation & Experiments
```powershell
# Run full evaluation on held-out test split
python -m src.deeprec.evaluation.evaluate

# Run benchmark experiments A through G (generates CSV and comparison plot)
python -m src.deeprec.evaluation.experiments --max_users 300
```

### Step 5: Run Unit Tests
```powershell
python -m pytest tests -v
```

### Step 6: Start REST API
```powershell
uvicorn src.deeprec.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation will be available at:  
👉 **http://127.0.0.1:8000/docs**

### Step 7: Launch Interactive Dashboard
```powershell
streamlit run dashboard/app.py
```
Dashboard will open at:  
👉 **http://localhost:8501**

---

## 7. REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service status, device, and artifact loading report. |
| `GET` | `/movies` | Paginated movie catalogue items with genre filters. |
| `GET` | `/recommendations/{user_id}` | Two-stage recommendations with candidate and ranking scores. |
| `POST` | `/interactions` | Record user interaction dynamically in the in-memory buffer. |
| `GET` | `/metrics` | Returns saved evaluation metrics and experiment comparisons. |

---

## 8. Limitations & Future Work

- **Duration Modeling:** MovieLens lacks continuous video watch durations; future extensions could incorporate session watch durations from streaming datasets.
- **ANN Serving:** For millions of catalogue items, integration with FAISS or ScaNN would replace exact matrix multiplication.
- **Online A/B Testing:** Offline metrics serve as a directional proxy; real-world efficacy requires live online interleaving or A/B testing.

See [`docs/limitations.md`](file:///docs/limitations.md) and [`docs/paper_implementation_mapping.md`](file:///docs/paper_implementation_mapping.md) for thorough discussions.
