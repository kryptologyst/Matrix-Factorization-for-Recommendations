# Matrix Factorization for Recommendations

A comprehensive, production-ready implementation of matrix factorization techniques for recommendation systems, featuring multiple algorithms, evaluation metrics, and an interactive demo.

## Overview

This project implements various matrix factorization methods for collaborative filtering, including:

- **Baselines**: Popularity-based, User-kNN, Item-kNN
- **Matrix Factorization**: SVD, NMF, ALS, BPR
- **Evaluation**: Comprehensive metrics including Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate@K, Coverage, Diversity, and Novelty
- **Demo**: Interactive Streamlit application for exploring recommendations

## Features

- Clean, modular codebase with type hints and comprehensive docstrings
- Multiple matrix factorization algorithms
- Comprehensive evaluation framework
- Interactive demo with Streamlit
- Synthetic data generation for testing
- Configurable training and evaluation pipeline
- Model leaderboard and comparison tools
- Production-ready project structure

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Matrix-Factorization-for-Recommendations.git
cd Matrix-Factorization-for-Recommendations
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

## Quick Start

### 1. Generate Synthetic Data

The project will automatically generate synthetic data if no real data is provided:

```bash
python scripts/train_evaluate.py --config configs/default.yaml
```

### 2. Train and Evaluate Models

```bash
python scripts/train_evaluate.py --config configs/default.yaml --output-dir results
```

### 3. Launch Interactive Demo

```bash
streamlit run demo.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   │   └── loader.py      # DataLoader and DataSplitter classes
│   ├── models/            # Recommendation models
│   │   └── matrix_factorization.py  # All model implementations
│   ├── evaluation/        # Evaluation metrics
│   │   └── metrics.py     # RecommendationMetrics and ModelLeaderboard
│   └── utils/             # Utility functions
├── configs/               # Configuration files
│   └── default.yaml       # Default configuration
├── data/                  # Data directory
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── scripts/               # Training and evaluation scripts
│   └── train_evaluate.py # Main training script
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks for analysis
├── assets/               # Static assets
├── demo.py               # Streamlit demo application
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## Data Format

The project expects the following data files in `data/raw/`:

### interactions.csv
```csv
user_id,item_id,rating,timestamp
0,1,5,1609459200
0,2,4,1609459300
...
```

### items.csv
```csv
item_id,title,category,year,rating_avg
0,Item 0,Action,2020,4.2
1,Item 1,Comedy,2019,3.8
...
```

### users.csv (optional)
```csv
user_id,age_group,gender,signup_date
0,18-25,M,1609459200
1,26-35,F,1609459300
...
```

## Models

### Baselines

- **PopularityRecommender**: Recommends most popular items
- **UserKNNRecommender**: User-based collaborative filtering
- **ItemKNNRecommender**: Item-based collaborative filtering

### Matrix Factorization

- **SVDRecommender**: Singular Value Decomposition
- **NMFRecommender**: Non-negative Matrix Factorization
- **ALSRecommender**: Alternating Least Squares (requires implicit library)
- **BPRRecommender**: Bayesian Personalized Ranking (requires implicit library)

## Configuration

The project uses YAML configuration files. See `configs/default.yaml` for available options:

```yaml
# Data configuration
data:
  test_size: 0.2
  val_size: 0.1
  split_method: "random"  # Options: "random", "temporal", "user_based"

# Model configuration
models:
  popularity: true
  user_knn: true
  user_knn_k: 50
  svd: true
  svd_factors: 50
  # ... more model configurations

# Evaluation configuration
evaluation:
  k_values: [5, 10, 20]
```

## Evaluation Metrics

The project provides comprehensive evaluation metrics:

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **Hit Rate@K**: Whether any relevant item appears in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Coverage**: Fraction of catalog items recommended
- **Diversity**: Average dissimilarity of recommendations
- **Novelty**: Average negative log popularity of recommendations

## Usage Examples

### Basic Training

```python
from src.data.loader import DataLoader, DataSplitter
from src.models.matrix_factorization import SVDRecommender
from src.evaluation.metrics import RecommendationMetrics

# Load data
data_loader = DataLoader("data")
interactions_df = data_loader.load_interactions()

# Split data
splitter = DataSplitter()
train_df, val_df, test_df = splitter.random_split(interactions_df)

# Train model
model = SVDRecommender(n_factors=50)
model.fit(train_df)

# Evaluate
metrics_calculator = RecommendationMetrics()
metrics = metrics_calculator.evaluate_model(model, test_df)
print(f"NDCG@10: {metrics['NDCG@10']:.4f}")
```

### Generate Recommendations

```python
# Generate recommendations for a user
recommendations = model.recommend(user_id="user_123", n_recommendations=10)
for item_id, score in recommendations:
    print(f"Item {item_id}: {score:.3f}")
```

### Model Comparison

```python
from src.evaluation.metrics import ModelLeaderboard

leaderboard = ModelLeaderboard()

# Add models to leaderboard
leaderboard.add_model("SVD", metrics_svd)
leaderboard.add_model("NMF", metrics_nmf)

# Get sorted leaderboard
results = leaderboard.get_leaderboard(metric="NDCG@10")
print(results)
```

## Interactive Demo

The Streamlit demo provides:

- **User Recommendations**: Generate recommendations for specific users
- **Item Similarity**: Find items similar to a given item
- **Model Comparison**: Compare performance across different models
- **Data Overview**: Explore dataset statistics and distributions

Launch the demo:
```bash
streamlit run demo.py
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Development

### Code Style

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **Type hints** for better code documentation
- **Google-style docstrings** for documentation

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
```

### Adding New Models

To add a new recommendation model:

1. Inherit from `BaseRecommender` in `src/models/matrix_factorization.py`
2. Implement required methods: `fit()`, `predict()`, `recommend()`
3. Add configuration options in `configs/default.yaml`
4. Update the model creation logic in `scripts/train_evaluate.py`

### Adding New Metrics

To add new evaluation metrics:

1. Add methods to `RecommendationMetrics` class in `src/evaluation/metrics.py`
2. Update the `evaluate_model()` method to include new metrics
3. Add configuration options if needed

## Performance Considerations

- **Memory Usage**: Large datasets may require chunked processing
- **Training Time**: ALS and BPR models can be slow on large datasets
- **Evaluation**: Use sampling for faster evaluation on large test sets
- **Caching**: Streamlit demo caches data and models for better performance

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce dataset size or use smaller models
3. **Slow Training**: Use fewer factors or iterations for faster training
4. **Demo Not Loading**: Check that data files exist in `data/raw/`

### Dependencies

Some models require additional libraries:
- **ALS/BPR**: Requires `implicit` library
- **SVD++**: Requires `surprise` library

Install missing dependencies:
```bash
pip install implicit surprise
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of scikit-learn, scipy, and other open-source libraries
- Inspired by the Netflix Prize and collaborative filtering research
- Uses best practices from the recommendation systems community

## References

- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems.
- Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian personalized ranking from implicit feedback.
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets.
# Matrix-Factorization-for-Recommendations
