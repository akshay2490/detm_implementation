# DETM Reproduction Fix List — UN General Debates

**Reference paper:** Dieng, Ruiz, Blei — *The Dynamic Embedded Topic Model* (2019)
**Reference implementation:** [adjidieng/DETM](https://github.com/adjidieng/DETM)

**Current symptoms:**
- Diversity ~0.11 (89% word overlap across topics)
- Coherence CV ~0.31, NPMI ~-0.18
- Perplexity 1.08 (should be in the hundreds)
- Global warming topic missing; all topic trajectories look identical

---

## Summary

| # | Severity | Fix | Location |
|---|----------|-----|----------|
| 1 | **CRITICAL** | Remove per-word normalization in reconstruction loss | `ReconstructionLoss.multinomial_loss()` — Cell 26 |
| 2 | **CRITICAL** | Adopt original's `.sum() * coeff` loss scaling | `DETM.forward()` — Cell 28 |
| 3 | **CRITICAL** | Enable trainable word embeddings | `Config.TRAIN_WORD_EMBEDDINGS` — Cell 4 |
| 4 | HIGH | Lower learning rate from 0.005 → 0.0001 | `Config.LEARNING_RATE` — Cell 4 |
| 5 | HIGH | Reduce LSTM η-encoder from 4×400 → 3×200 | `Config.LSTM_LAYERS/LSTM_HIDDEN` — Cell 4 |
| 6 | HIGH | Disable gradient clipping (original default) | `Config.CLIP_GRAD` — Cell 4 |
| 7 | HIGH | Add LR annealing (÷4 on plateau) | `DETMTrainer.train()` — Cell 32 |
| 8 | MEDIUM | Use Monte Carlo alpha KL (not analytical) | `DETM.compute_alpha_kl()` — Cell 28 |
| 9 | MEDIUM | Match alpha parameter shape (K,T,L) not (T,K,L) | `DETM.__init__()` — Cell 28 |
| 10 | MEDIUM | Initialize alpha logvar with randn, not zeros | `DETM.__init__()` — Cell 28 |
| 11 | LOW | Fix perplexity calculation | `TopicEvaluator.compute_perplexity()` — Cell 34 |
| 12 | LOW | Evaluate coherence across multiple time steps | `TopicEvaluator.evaluate_topics()` — Cell 34 |

---

## Critical Fixes — Primary Causes of Topic Collapse

### Fix #1: Reconstruction loss divided by document length (~50x weaker signal)

**Location:** `ReconstructionLoss.multinomial_loss()` — Cell 26

**Problem:** You divide NLL by document length, converting it to per-word NLL. The original sums over vocabulary and stops there. With avg ~50 tokens/doc, your reconstruction signal is ~50x weaker than KL terms, so the model minimizes KL by collapsing all topics to the prior (identical).

**Your code:**
```python
def multinomial_loss(self, bow, word_dist):
    word_dist = torch.clamp(word_dist, min=1e-10, max=1.0)
    doc_lengths = bow.sum(dim=-1)
    nll = -(bow * torch.log(word_dist)).sum(dim=-1)
    nll = nll / (doc_lengths + 1e-10)  # ← THIS KILLS YOUR MODEL
    return nll
```

**Original (`get_nll`):**
```python
loglik = torch.log(loglik + 1e-6)
nll = -loglik * bows
nll = nll.sum(-1)  # per-doc total NLL, no /doc_length
return nll
```

**Fix:**
```python
def multinomial_loss(self, bow, word_dist):
    word_dist = torch.clamp(word_dist, min=1e-10, max=1.0)
    nll = -(bow * torch.log(word_dist)).sum(dim=-1)
    return nll  # DELETE the doc_lengths division
```

---

### Fix #2: Loss scaling mismatch — terms on incompatible scales

**Location:** `DETM.forward()` — Cell 28, `ReconstructionLoss.forward()` — Cell 26

**Problem:** The original scales all terms to total-corpus magnitude using `coeff = num_docs / batch_size`. Both NLL and kl_theta are `.sum() * coeff`, while kl_eta and kl_alpha are already global sums. Your code mixes `.mean()` for recon/kl_theta with `/num_train_docs` for kl_eta/kl_alpha, putting terms on different scales.

**Original `forward()`:**
```python
coeff = num_docs / bsz
nll = nll.sum() * coeff
kl_theta = kl_theta.sum() * coeff
nelbo = nll + kl_alpha + kl_eta + kl_theta
```

**Fix — adopt the same convention:**

First, change `ReconstructionLoss.forward()` to return the per-sample tensor (not `.mean()`):

```python
def forward(self, bow, word_dist):
    if self.loss_type == 'multinomial':
        return self.multinomial_loss(bow, word_dist)  # (B,) — per-doc NLL
    ...
```

Then in `DETM.forward()`:

```python
if compute_loss:
    bsz = bow.shape[0]
    coeff = self.num_train_docs / bsz

    # 1. Reconstruction loss — sum over batch, scale to corpus
    recon_loss = self.recon_loss_fn(bow, word_dist)  # (B,)
    recon_loss = recon_loss.sum() * coeff

    # 2. KL theta — sum over batch, scale to corpus
    var_theta = theta_logvar.exp()
    kl_theta = 0.5 * torch.sum(
        (theta_mu - eta_batch).pow(2) + var_theta - theta_logvar - 1.0,
        dim=-1
    ).sum() * coeff  # .sum() not .mean()

    # 3. KL eta — already a global sum (no coeff needed)
    # kl_eta comes from temporal_baseline_encoder, already summed over T×K

    # 4. KL alpha — already a global sum (no coeff needed)
    kl_alpha = self.compute_alpha_kl()

    loss = recon_loss + kl_alpha + kl_eta + kl_theta
```

---

### Fix #3: Word embeddings frozen — original trains them

**Location:** `Config.TRAIN_WORD_EMBEDDINGS` — Cell 4

**Problem:** The original's default is `train_embeddings=1` (trainable). With frozen ρ, only α can shape topics. With trainable ρ, the model jointly adjusts the embedding space to make topics more separable.

**Fix:**
```python
# Config class
TRAIN_WORD_EMBEDDINGS = True  # was False
```

No architectural change needed — your `nn.Parameter(..., requires_grad=True)` is equivalent to the original's `nn.Linear(rho_size, vocab_size, bias=False)`.

---

## High Impact Fixes — Wrong Hyperparameters

### Fix #4: Learning rate 50x too high

**Location:** `Config.LEARNING_RATE` — Cell 4

**Problem:** The original README specifies `--lr 0.0001`. Your 0.005 with Adam causes the NaN batches you're guarding against and prevents stable convergence.

**Fix:**
```python
LEARNING_RATE = 0.0001  # was 0.005
```

---

### Fix #5: LSTM η-encoder oversized (4 layers/400 hidden vs original 3/200)

**Location:** `Config.LSTM_LAYERS`, `Config.LSTM_HIDDEN`, `Config.COMPRESSION_DIM` — Cell 4

**Problem:** The original uses `eta_nlayers=3, eta_hidden_size=200`. An oversized η-encoder absorbs signal that should go to topic embeddings α, making η explain everything and α irrelevant (all topics collapse). The temporal baselines should be a simple bias, not a powerful sequence model.

**Fix:**
```python
COMPRESSION_DIM = 200  # was 400
LSTM_LAYERS = 3        # was 4
LSTM_HIDDEN = 200      # was 400
```

---

### Fix #6: Gradient clipping enabled — original has it off

**Location:** `Config.CLIP_GRAD` — Cell 4, `DETMTrainer.train_epoch()` — Cell 32

**Problem:** The original defaults to `clip=0.0` (disabled). Your `CLIP_GRAD=2.0` caps gradient norms. Combined with the already-too-small reconstruction gradient (bug #1), this further suppresses the learning signal for topic differentiation.

**Fix:**
```python
CLIP_GRAD = 0.0  # was 2.0, matching original default
```

In `train_epoch()`, guard the clipping call:

```python
if self.config.CLIP_GRAD > 0:
    grad_norm = torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), self.config.CLIP_GRAD)
else:
    # Compute norm for logging but don't clip
    grad_norm = torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), float('inf'))
```

---

### Fix #7: No LR annealing — original divides LR by 4 on plateau

**Location:** `DETMTrainer.train()` — Cell 32

**Problem:** The original divides learning rate by 4 after validation loss doesn't improve for several epochs. Your trainer uses a constant LR with only early stopping. Without annealing, the model can't fine-tune topics in later epochs.

**Fix — add to `DETMTrainer.train()` inside the epoch loop, after checking `is_best`:**

```python
if not is_best:
    self.patience_counter += 1
    # Anneal LR every 10 epochs without improvement
    if self.patience_counter % 10 == 0:
        for pg in self.optimizer.param_groups:
            pg['lr'] /= 4.0
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"  LR annealed to {current_lr:.6f}")
```

---

## Medium Impact Fixes — Deviations from Original

### Fix #8: Alpha KL uses analytical expectation — original uses Monte Carlo sample

**Location:** `DETM.compute_alpha_kl()` — Cell 28

**Problem:** For t>0, you compute `E_{q(α_{t-1})}[KL(q(α_t)||p(α_t|α_{t-1}))]` analytically by adding `var_prev`. The original uses the **sampled** `α[t-1]` as the prior mean (single-sample Monte Carlo estimate).

Your analytical version adds an extra `var_prev / gamma_sq` per element. At initialization (logvar=0 → var=1.0, gamma_sq=0.005), this contributes ~200 per element × 735K elements = massive KL that over-regularizes α into the prior, crushing topic diversity.

**Original `get_alpha()`:**
```python
alphas[t] = self.reparameterize(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :])
p_mu_t = alphas[t-1]  # SAMPLED value, not mu_{t-1}
logsigma_p_t = torch.log(self.delta * torch.ones(...))
kl_t = self.get_kl(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :],
                    p_mu_t, logsigma_p_t)
```

**Fix — match original's Monte Carlo approach:**
```python
def compute_alpha_kl(self):
    gamma_sq = self.config.ALPHA_PRIOR_VARIANCE
    log_gamma = math.log(gamma_sq)

    # Sample alpha for all time steps
    alphas = []
    for t in range(self.num_time_steps):
        alpha_t = self._reparameterize_alpha(self.alpha_mu[t], self.alpha_logvar[t])
        alphas.append(alpha_t)

    total_kl = 0.0

    # t=0: prior is N(0, gamma_sq I)
    total_kl += 0.5 * torch.sum(
        self.alpha_logvar[0].exp() / gamma_sq
        + self.alpha_mu[0].pow(2) / gamma_sq
        - 1.0
        - self.alpha_logvar[0] + log_gamma
    )

    # t>0: prior mean is SAMPLED alpha[t-1], not mu[t-1]
    for t in range(1, self.num_time_steps):
        var_t = self.alpha_logvar[t].exp()
        total_kl += 0.5 * torch.sum(
            var_t / gamma_sq
            + (self.alpha_mu[t] - alphas[t-1]).pow(2) / gamma_sq
            - 1.0
            - self.alpha_logvar[t] + log_gamma
        )

    return total_kl

def _reparameterize_alpha(self, mu, logvar):
    if self.training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    return mu
```

---

### Fix #9: Alpha parameter shape is (T,K,L) — original is (K,T,L)

**Location:** `DETM.__init__()` — Cell 28

**Problem:** The original stores alpha as `(num_topics, num_times, rho_size)` = (K,T,L). Your code uses (T,K,L). While your indexing is internally consistent, the memory layout differs — with (K,T,L), the temporal dimension for each topic is contiguous, which may affect optimization dynamics. Not a correctness bug, but a fidelity deviation.

**Original:**
```python
self.mu_q_alpha = nn.Parameter(
    torch.randn(args.num_topics, args.num_times, args.rho_size))  # (K, T, L)
```

**Your code:**
```python
self.alpha_mu = nn.Parameter(
    torch.randn(num_time_steps, config.NUM_TOPICS, embedding_dim))  # (T, K, L)
```

**Fix:** If switching to (K,T,L), update all indexing throughout — `self.alpha_mu[:, t, :]` for time-step t, and reshape/permute when gathering by `time_indices` for a batch.

---

### Fix #10: Alpha logvar initialized to zeros — original uses randn

**Location:** `Config.INIT_ALPHA_LOGVAR`, `DETM.__init__()` — Cells 4, 28

**Problem:** The original initializes `logsigma_q_alpha` with `torch.randn` (random, mean 0, std 1). You use `torch.ones(...) * 0.0` (deterministic zeros). This means all alpha variances start at exactly 1.0. Combined with fix #8's analytical KL, this creates an enormous initial KL. With randn, some variances start above 1, some below, giving more diverse initial gradients.

**Fix:**
```python
self.alpha_logvar = nn.Parameter(
    torch.randn(num_time_steps, config.NUM_TOPICS, embedding_dim)
)
# Instead of: torch.ones(...) * config.INIT_ALPHA_LOGVAR
```

---

## Low Impact Fixes — Evaluation Bugs

### Fix #11: Perplexity calculation is wrong (gives 1.08 instead of ~hundreds)

**Location:** `TopicEvaluator.compute_perplexity()` — Cell 34

**Problem:** You accumulate `recon_loss.item() * len(bow)`, but `recon_loss` is already a per-word mean (bug #1). So you get `(per_word_NLL) × num_docs`, divide by `total_words` → tiny number → perplexity ≈ 1. After fixing #1+#2, `recon_loss` changes meaning, so compute NLL directly.

**Fix:**
```python
@staticmethod
def compute_perplexity(model, data_loader, device):
    model.eval()
    total_nll = 0.0
    total_words = 0
    with torch.no_grad():
        for batch in data_loader:
            bow = batch['bow'].to(device)
            time_indices = batch['time_idx'].to(device)
            output = model(bow, time_indices, compute_loss=False)
            # Compute NLL directly from word_dist and bow
            word_dist = torch.clamp(output['word_dist'], min=1e-10)
            nll = -(bow * torch.log(word_dist)).sum()  # total NLL for batch
            total_nll += nll.item()
            total_words += bow.sum().item()
    return np.exp(total_nll / total_words)
```

---

### Fix #12: Coherence evaluated only at last time step

**Location:** `TopicEvaluator.evaluate_topics()` → `DETM.get_topics()` — Cells 34, 28

**Problem:** `get_topics()` defaults to `time_idx=-1` (last time step, ~2015). Topic coherence may look different at earlier time steps. For a fair comparison with the paper, evaluate across multiple time steps.

**Fix:**
```python
def evaluate_topics(self, model, top_n_words=[10, 15, 20]):
    results = {}
    T = model.num_time_steps
    time_samples = [0, T//4, T//2, 3*T//4, T-1]

    for n in top_n_words:
        cv_scores = []
        npmi_scores = []
        for t in time_samples:
            topics_with_probs = model.get_topics(time_idx=t, top_n=n)
            topics = [[word for word, _ in topic] for topic in topics_with_probs]
            cv_scores.append(self.compute_coherence(topics, 'c_v'))
            npmi_scores.append(self.compute_coherence(topics, 'c_npmi'))

        results[f'coherence_cv_top{n}'] = np.mean(cv_scores)
        results[f'coherence_npmi_top{n}'] = np.mean(npmi_scores)

        # Diversity at last time step (standard reporting)
        topics_last = model.get_topics(time_idx=-1, top_n=n)
        topics_last = [[w for w, _ in t] for t in topics_last]
        results[f'diversity_top{n}'] = self.compute_topic_diversity(topics_last)

    return results
```

---

## After Applying All Fixes

1. **Delete** all cached checkpoints (`models/detm_best.pt`, `models/detm_epoch_*.pt`)
2. **Delete** cached preprocessed data (`data/preprocessed_data.pkl`) — not strictly necessary but good hygiene
3. **Retrain from scratch** with `LOAD_PRETRAINED = False`, `RESUME_TRAINING = False`
4. **Expected improvements:**
   - Diversity: 0.11 → 0.6+
   - Coherence CV: 0.31 → 0.45+
   - Perplexity: 1.08 → hundreds (correct range)
   - Distinct thematic topics including climate change / global warming should emerge
   - Topic trajectories should show meaningful temporal evolution (e.g., "terrorism" spiking post-2001)
