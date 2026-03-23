# DETM Update Plan — Matching Original Implementation

## Current Status

What's working well:
- Alpha logvar initialization (prior-centered) — massive improvement
- Reparameterize guards in DocumentTopicEncoder and TemporalBaselineEncoder
- KL formulas (standard normal at t=0, random walk at t>0)
- Reconstruction loss (no doc-length division)
- LR annealing, alpha diagnostics in TensorBoard

What still needs fixing (2 critical, 3 important, 2 minor):

---

## CRITICAL FIX 1: Loss scaling — adopt `.sum() * coeff`

### Problem
Your forward() uses `.mean()` for recon/kl_theta and `/num_train_docs` for kl_eta/kl_alpha. The original uses `.sum() * (D/B)` for recon/kl_theta and raw global sums for kl_eta/kl_alpha. While these are mathematically equivalent up to a constant, in practice **the `.mean()` convention produces very small gradients** that cause KL_θ to collapse monotonically (from 2.72 → 0.36 and falling). The encoder gives up on document-specific θ because the reconstruction gradient is too weak to overcome the KL_θ pull.

### What the original does (main.py + detm.py)

Training (`detm.py forward()`):
```python
coeff = num_docs / bsz          # D/B ≈ 57 for D=40000, B=700
nll = nll.sum() * coeff          # corpus-scale
kl_theta = kl_theta.sum() * coeff  # corpus-scale
nelbo = nll + kl_alpha + kl_eta + kl_theta  # all corpus-scale
```

Evaluation — **never calls forward()**. Uses completely separate functions:
```python
# _eta_helper: extracts mu_q_eta directly — no reparameterize
# get_theta: extracts mu_q_theta → softmax — no reparameterize
# alpha = model.mu_q_alpha — no sampling
# Then computes NLL / doc_length → per-word NLL → exp() for perplexity
```

### Changes needed

**Cell 26 — ReconstructionLoss.forward():**
Return per-sample tensor instead of mean, so DETM.forward() can apply its own scaling.

```python
def forward(self, bow, word_dist):
    """Returns per-document NLL — shape (B,). Caller handles aggregation."""
    if self.loss_type == 'multinomial':
        return self.multinomial_loss(bow, word_dist)  # (B,)
    elif self.loss_type == 'poisson':
        return self.poisson_loss(bow, word_dist)      # (B,)
    else:
        raise ValueError(f"Unknown loss type: {self.loss_type}")
```

**Cell 28 — DETM.forward(), the `if compute_loss:` block:**

```python
if compute_loss:
    bsz = bow.shape[0]
    coeff = self.num_train_docs / bsz

    # 1. Reconstruction: per-doc NLL → sum → scale to corpus
    recon_per_doc = self.recon_loss_fn(bow, word_dist)   # (B,)
    recon_loss = recon_per_doc.sum() * coeff

    # 2. KL theta: per-doc KL → sum → scale to corpus
    var_theta = theta_logvar.exp()
    kl_theta = 0.5 * torch.sum(
        (theta_mu - eta_batch).pow(2) + var_theta - theta_logvar - 1.0,
        dim=-1
    ).sum() * coeff

    # 3. KL eta: already a global sum over T×K — no scaling
    #    (kl_eta comes from temporal_baseline_encoder.forward())

    # 4. KL alpha: already a global sum over T×K×L — no scaling
    kl_alpha = self.compute_alpha_kl()

    # NELBO — all four terms on corpus scale
    loss = recon_loss + kl_theta + kl_eta + kl_alpha

    output.update({
        'loss': loss,
        'recon_loss': recon_loss,
        'kl_theta': kl_theta,
        'kl_eta': kl_eta,
        'kl_alpha': kl_alpha,
    })
```

Remove all `RECON_WEIGHT`, `KL_THETA_WEIGHT`, `KL_ETA_WEIGHT`, `KL_ALPHA_WEIGHT` — they're all 1.0 and the original doesn't use them.

**Cell 32 — DETMTrainer.validate():**
The original never calls forward() during eval. The simplest approach: keep calling forward() but use the same coeff. Since all reparameterize methods already return means during eval (model.eval()), the val loss will be deterministic and comparable across epochs. The coeff will be different (num_train_docs / val_batch_size), but that's fine — you're comparing val loss across epochs, not across train/val.

**Cell 34 — TopicEvaluator.compute_perplexity():**
Already correct — computes NLL directly from word_dist, independent of forward()'s scaling.

### Impact
This will amplify all gradients by ~57x. KL_θ gradient from reconstruction will become strong enough to push the encoder toward document-specific θ. You should see KL_θ **increase** (healthy) rather than collapse toward zero.

---

## CRITICAL FIX 2: Random data split (not sequential)

### Problem
Your data is sorted by year (preprocessing step 6), then `create_dataloaders` takes the first 80% as train. This means:
- Train: ~1970–2006
- Val: ~2006–2010
- Test: ~2010–2015

The model never trains on recent years. The original uses `np.random.permutation` so all time periods appear in all splits.

### What the original does (data_undebates.py)
```python
idx_permute = np.random.permutation(num_docs).astype(int)
# Then indexes into docs and timestamps using idx_permute
```

### Changes needed

**Cell 30 — create_dataloaders():**

```python
def create_dataloaders(processed_data, batch_size, train_split=0.85, val_split=0.05):
    bow_matrix = processed_data['bow_matrix']
    metadata = processed_data['metadata']
    time_indices = processed_data['temporal_info']['doc_to_time']

    n = len(bow_matrix)
    # Random permutation — matches original
    idx_permute = np.random.permutation(n)

    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_idx = idx_permute[:n_train]
    val_idx = idx_permute[n_train:n_train + n_val]
    test_idx = idx_permute[n_train + n_val:]

    train_dataset = DETMDataset(bow_matrix[train_idx], time_indices[train_idx])
    val_dataset = DETMDataset(bow_matrix[val_idx], time_indices[val_idx])
    test_dataset = DETMDataset(bow_matrix[test_idx], time_indices[test_idx])
    ...
```

Also change splits to 85/5/10 to match the original.

### Impact
All time periods represented in training. Model learns temporal dynamics across the full range.

---

## IMPORTANT FIX 3: RNN input — average raw counts, not normalized

### Problem
Your `_create_temporal_index()` L1-normalizes each document before averaging. The original averages raw counts (longer docs contribute proportionally more).

### What the original does (data.py `get_rnn_input`)
```python
for t in range(num_times):
    tmp = (times_batch == t).nonzero()
    docs = data_batch[tmp].squeeze().sum(0)   # SUM raw counts
    rnn_input[t] += docs
    cnt[t] += len(tmp)
rnn_input = rnn_input / cnt.unsqueeze(1)      # mean of raw counts
```

### Changes needed

**Cell 10 — DataPreprocessor._create_temporal_index():**

```python
# Replace:
normalized_docs = time_docs / (time_docs.sum(axis=1, keepdims=True) + 1e-10)
avg_bow_per_time[t_idx] = normalized_docs.mean(axis=0)

# With:
avg_bow_per_time[t_idx] = time_docs.mean(axis=0)  # mean of RAW counts
```

### Impact
LSTM gets stronger signal from longer, more informative documents. Matches original's η behavior.

---

## IMPORTANT FIX 4: Paragraph splitting — `'.\n'` not `'\n\n'`

### Problem
The original splits on `'.\n'` (period + newline). You split on `'\n\n'` (double newline). These produce different documents with different granularity.

### What the original does (data_undebates.py)
```python
if flag_split_by_paragraph:
    for dd, doc in enumerate(all_docs_ini):
        splitted_doc = doc.split('.\n')    # period + newline
        for ii in splitted_doc:
            docs.append(ii)
```

### Changes needed

**Cell 10 — DataPreprocessor.split_into_paragraphs():**
```python
def split_into_paragraphs(self, text: str) -> List[str]:
    # Original splits on '.\n' (sentence boundary in UN OCR text)
    paras = [p.strip() for p in text.split('.\n') if p.strip()]
    return paras if len(paras) > 1 else [text.strip()]
```

### Impact
Finer-grained documents with more focused vocabulary per paragraph. This is important for topic granularity — a paragraph about climate change won't be mixed with security vocabulary in the same BoW.

---

## IMPORTANT FIX 5: Min word length — `>1` not `>2`

### Problem
You filter `len(token) > 2` (removes 2-char words). Original uses `len(w) > 1` (keeps 2-char words like "un", "co", "us").

### Changes needed

**Cell 10 — DataPreprocessor.clean_text():**
```python
# Change:
if token.isalpha() and len(token) > 2 and token not in self.stopwords

# To:
if token.isalpha() and len(token) > 1 and token not in self.stopwords
```

### Impact
Minor — keeps some short words that may carry signal. Closer to original vocab.

---

## MINOR FIX 6: Build vocab from train only

### Problem
You build vocab from all documents, then split. Original splits first, then restricts vocab to words in train set.

### Changes needed
After implementing random split (Fix 2), build vocab from train documents only:

```python
# In pipeline (Cell 36):
# 1. Preprocess all docs (tokenize, paragraph split)
# 2. Random split into train/val/test indices
# 3. Build vocab from train docs only
# 4. Build BoW for ALL docs using train vocab
# 5. Train Word2Vec on train docs only
```

### Impact
Prevents test information leaking into vocabulary construction. More principled evaluation.

---

## MINOR FIX 7: Validation perplexity — per-word NLL matching original

### Problem
The original's `get_completion_ppl` divides NLL by doc length (per-word NLL) then takes mean across batch, then exp(). Your `compute_perplexity` sums total NLL across all docs and divides by total words. These are subtly different (arithmetic mean of per-word NLLs vs. global per-word NLL).

### What the original does (main.py `get_completion_ppl`)
```python
nll = nll.sum(-1)              # per-doc NLL
loss = nll / sums.squeeze()    # ÷ doc length → per-word NLL per doc
loss = loss.mean().item()      # MEAN of per-word NLLs across docs
ppl = math.exp(cur_loss)       # exp(mean of per-word NLLs)
```

### Changes needed
```python
@staticmethod
def compute_perplexity(model, data_loader, device):
    model.eval()
    acc_loss = 0.0
    cnt = 0
    with torch.no_grad():
        for batch in data_loader:
            bow = batch['bow'].to(device)
            time_indices = batch['time_idx'].to(device)
            output = model(bow, time_indices, compute_loss=False)
            word_dist = torch.clamp(output['word_dist'], min=1e-10)
            nll = -(bow * torch.log(word_dist)).sum(-1)       # (B,) per-doc NLL
            sums = bow.sum(-1)                                 # (B,) doc lengths
            per_word_nll = nll / (sums + 1e-10)               # (B,) per-word NLL
            acc_loss += per_word_nll.mean().item()             # mean over batch
            cnt += 1
    return float(math.exp(acc_loss / cnt))
```

### Impact
Makes perplexity numbers comparable with the paper.

---

## Implementation Order

1. **Fix 1 (loss scaling)** — most impactful for KL_θ collapse
2. **Fix 2 (random split)** — fixes data leakage
3. **Fix 3 (RNN input)** — fixes η quality
4. **Fix 4 (paragraph split)** — fixes document granularity
5. **Fix 5 (min word length)** — minor
6. **Fix 6 (train-only vocab)** — principled eval
7. **Fix 7 (perplexity)** — comparable metrics

**After all fixes: delete checkpoints, delete cached preprocessed data, retrain from scratch.**
