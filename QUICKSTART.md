# Quick Start Guide: DETM for UN General Debates

## Overview
This guide helps you get started with implementing the Dynamic Embedded Topic Model (DETM) on the UN General Debates dataset.

## Prerequisites
✅ Virtual environment activated: `source detm_env/bin/activate`  
✅ Dependencies installed: `pip install -r requirements.txt`  
✅ UN General Debates dataset downloaded to `data/`

## Step-by-Step Execution

### 1. Download Dataset

**Option A: Kaggle API (Recommended)**
```bash
# First, setup Kaggle credentials:
# 1. Create account at kaggle.com
# 2. Go to Account > API > Create New API Token
# 3. Place kaggle.json in ~/.kaggle/

cd ~/detm_topic_modelling
kaggle datasets download -d unitednations/un-general-debates
unzip un-general-debates.zip -d data/
```

**Option B: Manual Download**
1. Visit: https://www.kaggle.com/datasets/unitednations/un-general-debates
2. Download `un-general-debates.csv`
3. Place in `data/un-general-debates.csv`

### 2. Launch Jupyter Notebook

```bash
cd notebooks
jupyter notebook detm_un_debates.ipynb
```

### 3. Execute Notebook Cells

Open the notebook and run cells in order:

#### **Section 1-2: Setup** (Cells 1-2)
- Imports and configuration
- Sets random seeds for reproducibility
- Detects GPU if available

#### **Section 3: Data Preprocessing** (Cells 3-5)
```python
# Load data
df_raw = load_and_explore_data()

# Preprocess
preprocessor = DataPreprocessor(config)
processed_data = preprocessor.preprocess_corpus(df_raw)
```

**Expected Output:**
- Dataset statistics (documents, years, countries)
- Vocabulary size: ~5,000-10,000 words
- Document count after filtering
- Temporal coverage plot

#### **Section 4: Generate Embeddings** (Cell 6)
```python
# Generate word embeddings
embedding_gen = EmbeddingGenerator(config)
word_embeddings = embedding_gen.generate_vocabulary_embeddings(
    preprocessor.vocabulary
)
word_embeddings_tensor = torch.FloatTensor(word_embeddings)
```

**Expected Output:**
- Embedding model loading confirmation
- Embedding matrix shape: (vocab_size, 384)
- Progress bar for batch encoding

**⏱️ Time Estimate:** 2-5 minutes (depends on vocabulary size)

#### **Section 8: Create DataLoaders** (Cell 11)
```python
train_loader, val_loader, test_loader = create_dataloaders(
    processed_data,
    batch_size=config.BATCH_SIZE
)
```

**Expected Output:**
- Train/Val/Test split sizes (e.g., 80%/10%/10%)
- Number of documents in each split

#### **Section 7: Initialize Model** (Cell 10)
```python
model = DETM(config, word_embeddings_tensor)
model.idx2word = preprocessor.idx2word

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Expected Output:**
- Model architecture summary
- Total parameters: ~5-10M (depends on config)
- Device confirmation (CPU/GPU)

#### **Section 9: Train Model** (Cell 12)
```python
trainer = DETMTrainer(model, config, train_loader, val_loader, device)
history = trainer.train(num_epochs=50)  # Start with fewer epochs for testing
```

**Expected Output:**
- Training progress bar per epoch
- Loss values: Total, Reconstruction, KL
- Validation metrics
- Checkpoint saving messages
- Best model indicator

**⏱️ Time Estimate:** 
- CPU: 2-4 hours for 50 epochs
- GPU: 20-40 minutes for 50 epochs

**💡 Tips:**
- Start with 10-20 epochs to verify everything works
- Monitor for KL collapse (KL shouldn't go to zero)
- Reconstruction loss should decrease steadily

#### **Section 10-11: Evaluate** (Cells 15-16)
```python
# Compute metrics
evaluator = TopicEvaluator(
    processed_data['tokens_list'],
    preprocessor.vocabulary
)

metrics = evaluator.evaluate_topics(model, top_n_words=[10, 15, 20])
perplexity = evaluator.compute_perplexity(model, test_loader, device)

# Print results
print("\nEvaluation Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")
print(f"  Perplexity: {perplexity:.4f}")
```

**Expected Output:**
- C_v coherence: 0.4-0.6 (higher is better)
- C_npmi coherence: 0.0-0.3 (higher is better)
- Topic diversity: 0.7-0.95 (higher is better)
- Perplexity: 800-1500 (lower is better)

#### **Section 11: Visualize Topics** (Cells 17-19)
```python
# Topic word clouds
visualize_topics(model, num_topics_to_show=10, save_dir=config.OUTPUTS_DIR)

# Topic heatmap
plot_topic_heatmap(model, save_path=config.OUTPUTS_DIR / 'topic_heatmap.png')

# Training history
plot_training_history(history, save_path=config.OUTPUTS_DIR / 'training_history.png')
```

**Expected Output:**
- Word clouds for each topic
- Top-10 words printed per topic
- Topic-word probability heatmap
- Training curve plots (loss, KL, reconstruction)

#### **Section 13: Interactive Exploration** (Cell 22)
```python
create_topic_explorer(model, processed_data['metadata'])
```

**Expected Output:**
- Interactive widget with sliders
- Select topic ID to view words and word cloud
- Adjust top-N words dynamically

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in config
```python
config.BATCH_SIZE = 32  # or 16
```

### Issue: "KL divergence collapsed to zero"
**Solution:** Adjust KL weight
```python
config.KL_WEIGHT = 0.1  # Start lower, gradually increase
```

### Issue: "Poor topic quality (low coherence)"
**Solutions:**
1. Increase vocabulary size: `config.MAX_VOCAB_SIZE = 15000`
2. Adjust document frequency thresholds: `config.MIN_DF = 5`, `config.MAX_DF = 0.7`
3. Train longer: `config.NUM_EPOCHS = 150`
4. Try different number of topics: `config.NUM_TOPICS = 30` or `75`

### Issue: "Training is too slow"
**Solutions:**
1. Use GPU if available
2. Reduce vocabulary size: `config.MAX_VOCAB_SIZE = 5000`
3. Reduce batch size but increase learning rate proportionally
4. Use smaller embedding model: `config.EMBEDDING_MODEL = 'all-MiniLM-L6-v2'`

### Issue: "Topics are redundant (low diversity)"
**Solutions:**
1. Increase number of topics: `config.NUM_TOPICS = 75`
2. Adjust temporal variance: `config.TEMPORAL_VARIANCE = 0.01`
3. Add L2 regularization: `config.WEIGHT_DECAY = 1e-4`

## Configuration Recommendations

### For Quick Testing (Fast, Lower Quality)
```python
config.MAX_VOCAB_SIZE = 3000
config.NUM_TOPICS = 20
config.NUM_EPOCHS = 20
config.BATCH_SIZE = 128
config.MIN_DOC_LENGTH = 20
```

### For Paper Reproduction (Slow, High Quality)
```python
config.MAX_VOCAB_SIZE = 10000
config.NUM_TOPICS = 50
config.NUM_EPOCHS = 100
config.BATCH_SIZE = 64
config.MIN_DOC_LENGTH = 15
```

### For Large-Scale Experiments (Very Slow, Best Quality)
```python
config.MAX_VOCAB_SIZE = 15000
config.NUM_TOPICS = 100
config.NUM_EPOCHS = 200
config.BATCH_SIZE = 32
config.MIN_DOC_LENGTH = 10
```

## Expected Results

### Good Model Indicators:
✅ Reconstruction loss < 1000 at convergence  
✅ KL divergence in range 50-200 (not collapsed)  
✅ C_v coherence > 0.45  
✅ Topic diversity > 0.75  
✅ Semantically meaningful topics upon inspection  
✅ Topics show temporal evolution  

### Warning Signs:
⚠️ KL < 10 (posterior collapse)  
⚠️ Reconstruction loss increasing  
⚠️ Coherence < 0.3  
⚠️ All topics look similar (diversity < 0.5)  
⚠️ Topics contain mostly stopwords or rare words  

## Next Steps After Initial Run

1. **Save Best Model**
   ```python
   # Model is auto-saved to models/detm_best.pt
   # Load it later with:
   checkpoint = torch.load(config.MODELS_DIR / 'detm_best.pt')
   model.load_state_dict(checkpoint['model_state_dict'])
   ```

2. **Hyperparameter Tuning**
   - Experiment with different topic counts (25, 50, 75, 100)
   - Try different embedding models
   - Adjust learning rate and batch size

3. **Temporal Analysis**
   - Track topics across years
   - Identify emerging or declining topics
   - Correlate with historical events

4. **Country-Level Analysis**
   - Compare topic distributions by country
   - Identify regional patterns
   - Track policy focus changes

5. **Transfer to Financial Data**
   - Adapt preprocessing for financial text
   - Adjust vocabulary for domain-specific terms
   - Compare temporal patterns in financial events

## Saving and Loading Work

### Save Processed Data
```python
import pickle

# Save
with open('processed_data.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

# Load
with open('processed_data.pkl', 'rb') as f:
    processed_data = pickle.load(f)
```

### Save Results
```python
import json

results = {
    'config': config.__dict__,
    'metrics': metrics,
    'history': history
}

with open(config.OUTPUTS_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
```

## Getting Help

1. Check the main notebook documentation cells
2. Review the [README.md](../README.md)
3. Consult the original paper: Dieng et al. (2019)
4. Debug with smaller subset of data first

## Citation

```bibtex
@article{dieng2019topic,
  title={Topic modeling in embedding spaces},
  author={Dieng, Adji B and Ruiz, Francisco JR and Blei, David M},
  journal={Transactions of the Association for Computational Linguistics},
  volume={7},
  pages={439--453},
  year={2019}
}
```

---

**Ready to start?** Run `jupyter notebook detm_un_debates.ipynb` and follow along! 🚀
