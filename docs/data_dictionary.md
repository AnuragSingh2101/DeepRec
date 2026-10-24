# Data Dictionary — DeepRec

This document describes all raw files, processed schemas, feature mappings, and metadata schemas in DeepRec.

---

## 1. Raw Dataset Files (`data/raw/ml-100k/`)

### A. `u.data` (Ratings)
- **Format:** Tab-separated (`\t`), no header.
- **Rows:** 100,000 interactions.
- **Columns:**
  1. `user_id` (int): User ID (1 to 943).
  2. `movie_id` (int): Movie ID (1 to 1682).
  3. `rating` (int): Explicit rating on 1–5 scale.
  4. `timestamp` (int): Unix epoch seconds.

### B. `u.item` (Movies)
- **Format:** Pipe-separated (`|`), Latin-1 encoded, no header.
- **Rows:** 1,682 movies.
- **Columns:**
  - `movie_id` (int): Unique raw identifier.
  - `title` (str): Movie title and release year in parentheses.
  - `release_date` (str): e.g., `"01-Jan-1995"`.
  - `video_release_date` (str): Empty / unused.
  - `imdb_url` (str): IMDB URL.
  - 19 binary genre columns: `unknown`, `Action`, `Adventure`, `Animation`, `Children's`, `Comedy`, `Crime`, `Documentary`, `Drama`, `Fantasy`, `Film-Noir`, `Horror`, `Musical`, `Mystery`, `Romance`, `Sci-Fi`, `Thriller`, `War`, `Western`.

### C. `u.user` (Users)
- **Format:** Pipe-separated (`|`), no header.
- **Rows:** 943 users.
- **Columns:**
  - `user_id` (int): 1 to 943.
  - `age` (int): User age in years.
  - `gender` (str): `"M"` or `"F"`.
  - `occupation` (str): e.g., `"administrator"`, `"engineer"`, `"student"`.
  - `zip_code` (str): US postal code.

---

## 2. Processed Files (`data/processed/`)

| File | Description | Key Columns |
| :--- | :--- | :--- |
| `train.csv` | Chronological training interactions per user. | `user_idx`, `movie_idx`, `rating`, `timestamp`, `position` |
| `validation.csv` | Second-to-last chronological interaction per user. | `user_idx`, `movie_idx`, `rating`, `timestamp` |
| `test.csv` | Final chronological interaction per user. | `user_idx`, `movie_idx`, `rating`, `timestamp` |
| `movies.csv` | Catalogue with extracted release year and genre indicators. | `movie_idx`, `movie_id`, `title`, `release_year`, 19 genre flags |
| `users.csv` | User demographic features with mapped categorical codes. | `user_idx`, `age`, `gender_code`, `occupation_code` |
| `user_map.json` | Deterministic map: `raw_user_id` $\to$ `user_idx`. | Key: raw ID, Value: 1-indexed contiguous index. |
| `movie_map.json` | Deterministic map: `raw_movie_id` $\to$ `movie_idx`. | Index `0` is strictly reserved for padding. |
| `metadata.json` | Global catalog sizes, splits, and token definitions. | `num_users`, `num_movies`, `padding_idx=0` |

---

## 3. Continuous Ranking Feature Schema

The Deep Ranking Model takes 27 continuous features per candidate pair ($9 \text{ base features} \times 3 \text{ powers: } x, x^2, \sqrt{x}$):

| Base Feature | Formula / Extraction Source | Description |
| :--- | :--- | :--- |
| `candidate_score` | Stage 1 dot product output $s(u, v) = u^\top v + b$ | Retrieval confidence from candidate generation network. |
| `item_popularity` | $\text{count}_{\text{train}}(i) / \max(\text{counts})$ | Normalized interaction frequency in training split. |
| `item_avg_rating` | Bayesian smoothed mean rating | Dampened mean: $\frac{N \cdot \bar{r} + 5 \cdot \mu}{N + 5}$. |
| `item_rating_count_log` | $\log(1 + N_{\text{ratings}})$ | Scale-damped item interaction volume. |
| `user_mean_rating` | $\frac{1}{|I_u|} \sum_{i \in I_u} r_{u,i}$ | User's rating generosity / strictness baseline. |
| `user_count_log` | $\log(1 + |I_u|)$ | User activity level indicator. |
| `genre_affinity` | $\frac{u_{\text{genres}} \cdot m_{\text{genres}}}{\|u\| \|m\|}$ | Cosine similarity between user watched genres and candidate genres. |
| `interaction_recency` | $\log(1 + \Delta_{\text{days}})$ | Time elapsed since user's most recent interaction. |
| `release_year` | $(\text{year} - 1920) / 80.0$ | Normalized film age relative to modern catalogue. |
