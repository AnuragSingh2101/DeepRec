# Project Limitations & Design Trade-offs — DeepRec

This document transparently describes the limitations, architectural trade-offs, and boundary conditions of the DeepRec implementation relative to production recommendation systems and the original research paper (*Covington et al., 2016*).

---

## 1. Dataset Limitations: MovieLens 100K vs. YouTube Production Logs

1. **Implicit Watch Signals vs. Explicit Ratings:**
   - In YouTube, the primary input signal is continuous streaming telemetry: video impressions, user clicks, play starts, completion percentages, and watch durations.
   - MovieLens 100K consists of voluntary, explicit 1-to-5 star ratings. Ratings reflect reflective evaluation after consumption rather than immediate attention capture.
   - DeepRec formulates recommendation as predicting the next interacted movie chronologically, using ratings $\ge 4.0$ as an engagement proxy.

2. **Absence of Observed Watch Durations:**
   - Covington et al. formulated the ranking objective as **Weighted Logistic Regression**, where positive instances are weighted by observed continuous watch time $T$, so that the learned odds $\frac{p}{1-p}$ approximate expected watch duration $\mathbb{E}[T]$.
   - Because MovieLens lacks duration logs, DeepRec avoids fabricating synthetic watch durations. Instead, DeepRec adopts Binary Cross-Entropy with Logits on high-engagement ratings.

3. **Absence of Negative Impression Logs:**
   - In production systems, negative examples are drawn from unclicked impressions that were actually shown on screen to the user.
   - In MovieLens, unobserved user-item pairs represent a mix of items the user disliked and items the user was never exposed to. DeepRec samples unchosen candidates to simulate impression logs.

4. **Corpus Scale:**
   - YouTube operates over tens of millions of active video items and billions of users, necessitating distributed parameter servers and hierarchical cluster indexing.
   - MovieLens 100K contains 943 users and 1,682 movies. This scale permits exact vectorized retrieval in single-digit milliseconds on a standard laptop CPU without approximate indexing distortion.

---

## 2. Offline Metrics vs. Online Engagement

1. **Exposure Bias:**
   - Offline evaluation metrics (Recall@K, NDCG@K) evaluate whether the model recovers items the user previously found and rated under historical interface conditions.
   - A model that discovers novel, high-quality items will be penalized offline if the user never rated them in the historical dataset.

2. **Serendipity vs. Relevance:**
   - Highly accurate candidate retrieval often favors popular or closely related franchise items. Offline metrics do not measure filter bubble effects or user boredom.

---

## 3. Serving & Infrastructure Differences

1. **Single-Node In-Memory vs. Distributed RPC:**
   - Production YouTube recommendation uses an asynchronous multi-stage pipeline across thousands of worker nodes with gRPC interfaces and sub-10ms strict SLAs.
   - DeepRec runs as a single-node Python service orchestrated via FastAPI and Streamlit, suitable for local reproducibility, academic portfolios, and workstation evaluation.

2. **Approximate Nearest Neighbor (ANN) vs. Vectorized Dot Product:**
   - For millions of items, hierarchical k-means or HNSW indexing is essential.
   - In DeepRec, candidate retrieval is computed via batched PyTorch matrix multiplication (`torch.matmul(u, V.T)`), which guarantees exact top-K recall without index compression loss.
