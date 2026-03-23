# Systematic Comparison: Original DETM vs Your Implementation

## 1. Preprocessing (`scripts/data_undebates.py` vs `DataPreprocessor`)

### 1.1 Paragraph splitting

| Aspect | Original | Yours | Match? |
|--------|----------|-------|--------|
| Split method | `doc.split('.\n')` (period + newline) | `text.split('\n\n')` (double newline) | **DIFFERENT** |
| Fallback | No fallback — always splits on `'.\n'` | Falls back to full speech if no `\n\n` | **DIFFERENT** |

This is a significant difference. The original splits on **sentence-ending period followed by newline** (`'.\n'`), not on double newlines. This produces different (and generally more) paragraph-documents. Many UN speeches have `.\n` at the end of every sentence (OCR formatting), so the original gets finer-grained documents.

### 1.2 Text cleaning

| Aspect | Original | Yours | Match? |
|--------|----------|-------|--------|
| Lowercase | Yes | Yes | ✓ |
| Apostrophe handling | Replaces `'` and `'` with space | NLTK `word_tokenize` handles internally | **DIFFERENT** |
| Punctuation removal | `str.translate(maketrans(..., punctuation + "0123456789"))` | `token.isalpha()` filter | Functionally similar ✓ |
| Min word length | `len(w) > 1` (>1 char) | `len(token) > 2` (>2 chars) | **DIFFERENT** — you discard 2-letter words |
| Tokenizer | Simple `.split()` on whitespace | NLTK `word_tokenize` | **DIFFERENT** |
| Stopwords | External `stops.txt` file (~600 words) | NLTK English + 18 UN words | **DIFFERENT** — different stopword lists |
| Lemmatization | None | None | ✓ |
| Number removal | Yes (digits stripped from strings) | Yes (isalpha filter) | ✓ |

### 1.3 Vocabulary building

| Aspect | Original | Yours | Match? |
|--------|----------|-------|--------|
| min_df | 10 | 10 | ✓ |
| max_df | 0.7 | 0.7 | ✓ |
| Method | `CountVectorizer(min_df, max_df, stop_words=None)` | Manual doc-frequency counting | Functionally equivalent ✓ |
| Stopwords in CV | `stop_words=None` (stopwords removed earlier) | Stopwords removed during tokenization | ✓ |
| Post-filter | Removes stopwords from vocab list, then removes words not in train set | No post-filter for train-only words | **DIFFERENT** |
| Sort order | Sorted by sum of counts ascending | Sorted alphabetically | Irrelevant to model |

**Critical difference:** The original rebuilds the vocabulary after splitting into train/test, keeping **only words that appear in the training set**. Your code builds vocabulary from all documents before splitting. This means your vocab may contain words that only appear in val/test, which wastes vocab slots and slightly changes the embedding space.

### 1.4 Train/val/test split

| Aspect | Original | Yours | Match? |
|--------|----------|-------|--------|
| Train fraction | 85% | 80% | **DIFFERENT** |
| Test fraction | 10% | 10% | ✓ |
| Val fraction | 5% | 10% | **DIFFERENT** |
| Split method | Random permutation (`np.random.permutation`) | Sequential (first 80% = train) | **DIFFERENT** |
| Test split for perplexity | Test docs split in 2 halves (h1/h2) | No half-split | **DIFFERENT** |

**The sequential split is problematic.** Your data is sorted by year (step 6 in `preprocess_corpus`), then you take the first 80% as train. This means train = early years, test = most recent years. The original uses random permutation so all time periods are represented in all splits. Your model never sees recent years during training.

---

## 2. Model (`detm.py` — Original vs Yours)

### 2.1 Encoder (theta inference)

| Aspect | Original | Yours | Match? |
|--------|----------|-------|--------|
| Input | `[normalized_bow, eta_td]` | `[bow_norm, eta]` | ✓ |
| BoW normalization | `data_batch / sums` (L1 norm) — done in main.py, passed as `normalized_bows` | `bow / bow.sum(dim=1, keepdim=True)` — done inside encoder | ✓ |
| Architecture | Linear→act→Linear→act→μ,logσ | Linear→ReLU→Linear→ReLU→μ,logvar | ✓ |
| Activation | Configurable (default: `relu`) | Hardcoded ReLU | ✓ (default matches) |
| Dropout | Configurable `enc_drop` (default 0.0) | 0.0 | ✓ |
| Reparameterize | Returns μ during eval | **Always samples (BUG)** | **DIFFERENT** |

### 2.2 Decoder (beta computation)

| Aspect | Original | Yours | Match? |
|--------|----------|-------|--------|
| Word embeddings (trainable) | `nn.Linear(rho_size, vocab_size, bias=False)` when trainable | `nn.Parameter(requires_grad=True)` | Mathematically equivalent ✓ |
| Word embeddings (frozen) | `nn.Embedding` → extract `.weight.data` as fixed tensor | `nn.Parameter(requires_grad=False)` | ✓ |
| Beta formula | `softmax(alpha @ rho.T)` | `softmax(alpha @ embeddings.T)` | ✓ |

### 2.3 Temporal baseline (eta inference)

| Aspect | Original | Yours | Match? |
|--------|----------|-------|--------|
| Compression | `nn.Linear(vocab_size, eta_hidden_size)` | `nn.Linear(vocab_size, compression_dim)` | ✓ |
| LSTM layers | 3 (default `eta_nlayers`) | 3 | ✓ |
| LSTM hidden | 200 (default `eta_hidden_size`) | 200 | ✓ |
| Output input | `[lstm_output_t, eta_{t-1}]` | `[lstm_output_t, eta_prev]` | ✓ |
| Reparameterize | Returns μ during eval | **Always samples (BUG)** | **DIFFERENT** |
| RNN input | `get_rnn_input()` averages **raw counts** per time step (sum of counts / num docs per time) | `avg_bow_per_time` averages **L1-normalized** docs then means | **DIFFERENT** |

**RNN input difference is important.** The original's `get_rnn_input`:
```python
# For each timestep, SUM all raw BoW vectors, then divide by count
docs = data_batch[tmp].squeeze().sum(0)  # sum raw counts
rnn_input[t] += docs
cnt[t] += len(tmp)
rnn_input = rnn_input / cnt.unsqueeze(1)  # mean of raw counts
```

Your code in `_create_temporal_index`:
```python
# Normalize each doc first, THEN average
normalized_docs = time_docs / (time_docs.sum(axis=1, keepdims=True) + 1e-10)
avg_bow_per_time[t_idx] = normalized_docs.mean(axis=0)
```

The original averages raw counts (so longer documents contribute more). Yours L1-normalizes first (so all documents contribute equally regardless of length). These produce different LSTM inputs.

### 2.4 Alpha (topic embeddings)

| Aspect | Original | Yours | Match? |
|--------|----------|-------|--------|
| Shape | `(K, T, L)` | `(T, K, L)` | **DIFFERENT** (transposes) |
| Mu init | `torch.randn(K, T, L)` | `torch.randn(T, K, L) * 1.0` | ✓ (same distribution) |
| Logvar init | `torch.randn(K, T, L)` | `torch.randn(T, K, L)` | ✓ |
| Sampling | One sample per time step, shared across batch | One sample per unique time step in batch | ✓ |

### 2.5 Forward pass / loss computation

| Aspect | Original | Yours | Match? |
|--------|----------|-------|--------|
| **NLL input** | **Raw BoW counts (`data_batch`)** | **Raw BoW counts (`bow`)** | ✓ |
| **Theta input** | **Normalized BoW (`normalized_data_batch`)** | **Normalized inside encoder** | ✓ |
| NLL formula | `-(log(theta @ beta + 1e-6) * bows).sum(-1)` | `-(bow * log(clamp(word_dist, 1e-10))).sum(-1)` | ✓ |
| NLL scaling | `.sum() * coeff` where `coeff = D/B` | `.mean()` | Equivalent up to constant |
| KL_theta scaling | `.sum() * coeff` | `.mean()` | Equivalent up to constant |
| KL_theta prior mean | `eta_td` (η is prior mean for θ) | `eta_batch` | ✓ |
| KL_theta prior var | `zeros` → variance = 1.0 (standard logistic-normal) | `1.0` (implicit in formula) | ✓ |
| KL_eta prior at t=0 | `logsigma_p_0 = zeros` → **variance = 1.0** | Now fixed to use variance 1.0 | ✓ |
| KL_alpha prior at t=0 | `logsigma_p_0 = zeros` → **variance = 1.0** | Now fixed to use variance 1.0 | ✓ |
| KL_eta/alpha scaling | Raw global sum (no division) | `/num_train_docs` | Equivalent up to constant |

### 2.6 Key finding: the original passes TWO different BoW tensors

This is subtle but critical. Look at the original's training loop in `main.py`:

```python
data_batch, times_batch = data.get_batch(...)  # RAW counts
sums = data_batch.sum(1).unsqueeze(1)
if args.bow_norm:
    normalized_data_batch = data_batch / sums    # L1-normalized
else:
    normalized_data_batch = data_batch

loss, nll, kl_alpha, kl_eta, kl_theta = model(
    data_batch,              # RAW counts → used for NLL computation
    normalized_data_batch,   # NORMALIZED → fed into theta encoder
    times_batch, train_rnn_inp, args.num_docs_train)
```

And inside `detm.py`:
```python
def forward(self, bows, normalized_bows, times, rnn_inp, num_docs):
    ...
    theta, kl_theta = self.get_theta(eta, normalized_bows, times)  # encoder gets NORMALIZED
    ...
    nll = self.get_nll(theta, beta, bows)  # NLL uses RAW counts
```

**Your code does this correctly** — the encoder normalizes internally, and the NLL uses raw `bow`. Just noting this matches.

---

## 3. Data loading (`data.py` vs `DETMDataset`)

### 3.1 Data format

| Aspect | Original | Yours | Match? |
|--------|----------|-------|--------|
| Storage format | Sparse (token indices + counts in separate .mat files) | Dense numpy matrix | Different format, same content ✓ |
| Batch construction | `get_batch()` builds dense matrix on-the-fly from sparse | Pre-built dense tensor | ✓ (equivalent) |

### 3.2 RNN input construction

| Aspect | Original | Yours | Match? |
|--------|----------|-------|--------|
| Function | `get_rnn_input()` in data.py | `_create_temporal_index()` in DataPreprocessor | **DIFFERENT** |
| Normalization | Average of raw counts per time step | Average of L1-normalized docs per time step | **DIFFERENT** |

### 3.3 Test set evaluation (perplexity)

| Aspect | Original | Yours | Match? |
|--------|----------|-------|--------|
| Method | Split each test doc into 2 halves; use h1 for θ estimation, h2 for NLL | Use full test docs for both | **DIFFERENT** |

The original's perplexity is computed as:
1. Take first half of each test doc → infer θ
2. Take second half → compute NLL using that θ
3. Perplexity = exp(NLL_h2 / total_words_h2)

This is a proper held-out evaluation that prevents the model from "seeing" the words it's being evaluated on. Your method uses the same words for both θ inference and NLL computation, which gives optimistically low perplexity.

---

## 4. Summary: Remaining issues causing the train/val recon gap

### Definite bugs (fix these):

1. **Reparameterize always samples in DocumentTopicEncoder and TemporalBaselineEncoder** — during eval, θ and η still get noise. Alpha correctly returns the mean. Fix: add `if self.training` guard.

2. **Sequential data split** — your data is sorted by year, so train = 1970–2005, val = 2005–2010, test = 2010–2015 (approximately). The model has never seen recent-year documents. Fix: random permutation before splitting.

3. **RNN input normalization** — original averages raw counts (length-weighted), you average L1-normalized docs (length-invariant). This changes the LSTM's input distribution.

### Deviations (may affect topic quality):

4. **Paragraph splitting** — `'.\n'` (original) vs `'\n\n'` (yours) produces different document granularity.

5. **Min word length** — `>1` char (original) vs `>2` chars (yours). You're dropping all 2-letter words.

6. **Stopwords** — different stopword lists.

7. **Vocabulary built from all data** — original restricts to train-only words after splitting.

8. **Train split ratio** — 85% (original) vs 80% (yours).

9. **Perplexity evaluation** — original uses held-out half-document method.
