# Dynamic Embedded Topic Model (DETM) Implementation

Implementation of the Dynamic Embedded Topic Model from Dieng et al. (2019) "Topic Modeling in Embedding Spaces" applied to the UN General Debates corpus.

## Project Structure

```
detm_topic_modelling/
├── data/                   # Data directory
│   ├── un-general-debates.csv (download from Kaggle)
│   ├── preprocessed_data.pkl
│   └── word_embeddings.npy
├── models/                 # Saved model checkpoints
│   ├── detm_best.pt
│   └── detm_epoch_*.pt
├── outputs/                # Results and visualizations
│   ├── training_history.png
│   ├── topic_wordclouds.png
│   ├── topic_heatmap.png
│   └── results.json
├── notebooks/              # Jupyter notebooks
│   └── detm_un_debates.ipynb (main implementation)
├── detm_env/               # Virtual environment
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup Instructions

### 1. Activate Virtual Environment

```bash
source detm_env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download UN General Debates Dataset

**Option A: Using Kaggle API (Recommended)**
```bash
# Setup Kaggle credentials first (place kaggle.json in ~/.kaggle/)
kaggle datasets download -d unitednations/un-general-debates
unzip un-general-debates.zip -d data/
```

**Option B: Manual Download**
1. Go to: https://www.kaggle.com/datasets/unitednations/un-general-debates
2. Download the CSV file
3. Place it in `data/un-general-debates.csv`

### 4. Launch Jupyter Notebook

```bash
cd notebooks
jupyter notebook detm_un_debates.ipynb
```

## Model Architecture

### Generative Process
- **Topic embeddings** evolve over time: α_t ~ N(α_{t-1}, σ²I)
- **Topic-word distributions**: β_k,t = softmax(α_k,t · ρ) where ρ are word embeddings
- **Document topic proportions**: θ_t ~ N(0, I)
- **Word generation**: w ~ Categorical(softmax(β^T θ))

### Variational Inference
- **Approximate posterior**: q(θ_t | w_t) = N(μ_t, Σ_t)
- **Encoder network** outputs μ_t and log σ_t from bag-of-words
- **ELBO objective**: E_q[log p(w|θ,α,ρ)] - KL(q(θ)||p(θ))

## Modular Components

### 1. Data Preprocessing (`DataPreprocessor`)
- Text cleaning and tokenization
- Vocabulary building with frequency thresholds
- Bag-of-words representation
- Temporal ordering

### 2. Embeddings (`EmbeddingGenerator`)
- BERT/Sentence Transformer word embeddings
- Batch processing for efficiency
- Vocabulary-aligned embedding matrix

### 3. Model Architecture
- **Encoder** (`DETMEncoder`): Inference network q(θ|w)
- **Decoder** (`DETMDecoder`): Generative model p(w|θ,α,ρ)
- **Temporal** (`TemporalTransition`): Topic evolution modeling

### 4. Loss Modules (Modular Design)
- **KL Divergence** (`KLDivergenceLoss`): Separate KL computation
- **Reconstruction** (`ReconstructionLoss`): Word likelihood computation
- Clear algorithm boundaries for each loss component

### 5. Training (`DETMTrainer`)
- ELBO optimization
- Learning rate scheduling
- Checkpointing and early stopping
- Loss monitoring

### 6. Evaluation (`TopicEvaluator`)
- Topic coherence (C_v, C_npmi)
- Topic diversity
- Perplexity calculation

### 7. Visualization
- Training history plots
- Topic word clouds
- Topic-word heatmaps
- Interactive exploration widgets

## Configuration

Key hyperparameters in `Config` class:

```python
NUM_TOPICS = 50              # Number of topics (K)
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Sentence transformer
HIDDEN_DIM = 512             # Encoder hidden dimension
NUM_ENCODER_LAYERS = 2       # Encoder MLP layers
DROPOUT = 0.2                # Dropout rate

BATCH_SIZE = 64              # Training batch size
NUM_EPOCHS = 100             # Maximum epochs
LEARNING_RATE = 1e-3         # Initial learning rate
CLIP_GRAD = 5.0              # Gradient clipping

MIN_DF = 10                  # Minimum document frequency
MAX_DF = 0.5                 # Maximum document frequency
MAX_VOCAB_SIZE = 10000       # Maximum vocabulary size
```

## Usage

### Basic Workflow

```python
# 1. Load and preprocess data
df_raw = load_and_explore_data()
preprocessor = DataPreprocessor(config)
processed_data = preprocessor.preprocess_corpus(df_raw)

# 2. Generate embeddings
embedding_gen = EmbeddingGenerator(config)
word_embeddings = embedding_gen.generate_vocabulary_embeddings(preprocessor.vocabulary)
word_embeddings_tensor = torch.FloatTensor(word_embeddings)

# 3. Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    processed_data, batch_size=config.BATCH_SIZE
)

# 4. Initialize and train model
model = DETM(config, word_embeddings_tensor)
model.idx2word = preprocessor.idx2word

trainer = DETMTrainer(model, config, train_loader, val_loader, device)
history = trainer.train()

# 5. Evaluate
evaluator = TopicEvaluator(processed_data['tokens_list'], preprocessor.vocabulary)
metrics = evaluator.evaluate_topics(model)
perplexity = evaluator.compute_perplexity(model, test_loader, device)

# 6. Visualize
visualize_topics(model, save_dir=config.OUTPUTS_DIR)
plot_training_history(history)
```

## Evaluation Metrics

### Topic Coherence
- **C_v**: Measures semantic coherence using word co-occurrence
- **C_npmi**: Normalized pointwise mutual information

### Topic Diversity
- Percentage of unique words in top-N words across topics
- Higher is better (less redundancy)

### Perplexity
- Measures predictive performance on held-out data
- Lower is better

## Paper Comparison

### Original Paper (Dieng et al. 2019)
- Used Word2Vec embeddings
- Tested on: 20 Newsgroups, Reuters, UN Debates
- Reported coherence improvements over LDA and static ETM

### This Implementation
- Uses BERT/Sentence Transformer embeddings (configurable)
- Applied to full UN General Debates corpus
- Modular design for easy experimentation
- Comprehensive evaluation and visualization

## Extensions and Next Steps

1. **Temporal Analysis**: Track topic evolution year-by-year
2. **Country-level Analysis**: Compare topic distributions across countries
3. **Hyperparameter Tuning**: Grid search for optimal configurations
4. **Baseline Comparisons**: LDA, Static ETM benchmarks
5. **Transfer to Financial Data**: Adapt for financial text applications

## Dependencies

Core libraries:
- `torch >= 2.0.0` - Deep learning framework
- `transformers >= 4.30.0` - BERT models
- `sentence-transformers >= 2.2.0` - Sentence embeddings
- `gensim >= 4.3.0` - Topic coherence metrics
- `pandas`, `numpy`, `scipy` - Data processing
- `matplotlib`, `seaborn`, `wordcloud` - Visualization

## Citation

If using this implementation, please cite the original paper:

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

## License

This implementation is for research and educational purposes.

## Contact

For questions or issues, please check the notebook documentation or raise an issue.
