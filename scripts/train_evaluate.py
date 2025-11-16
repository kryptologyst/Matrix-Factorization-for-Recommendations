"""Main training and evaluation script for matrix factorization recommendations."""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.loader import DataLoader, DataSplitter, set_seed
from models.matrix_factorization import (
    PopularityRecommender, UserKNNRecommender, ItemKNNRecommender,
    SVDRecommender, NMFRecommender, ALSRecommender, BPRRecommender
)
from evaluation.metrics import RecommendationMetrics, ModelLeaderboard

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_models(config: Dict) -> List:
    """Create recommendation models based on configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        List of model instances.
    """
    models = []
    
    # Baselines
    if config.get("models", {}).get("popularity", True):
        models.append(PopularityRecommender())
    
    if config.get("models", {}).get("user_knn", True):
        k = config.get("models", {}).get("user_knn_k", 50)
        models.append(UserKNNRecommender(k=k))
    
    if config.get("models", {}).get("item_knn", True):
        k = config.get("models", {}).get("item_knn_k", 50)
        models.append(ItemKNNRecommender(k=k))
    
    # Matrix Factorization models
    if config.get("models", {}).get("svd", True):
        n_factors = config.get("models", {}).get("svd_factors", 50)
        models.append(SVDRecommender(n_factors=n_factors))
    
    if config.get("models", {}).get("nmf", True):
        n_factors = config.get("models", {}).get("nmf_factors", 50)
        models.append(NMFRecommender(n_factors=n_factors))
    
    if config.get("models", {}).get("als", True):
        n_factors = config.get("models", {}).get("als_factors", 50)
        models.append(ALSRecommender(n_factors=n_factors))
    
    if config.get("models", {}).get("bpr", True):
        n_factors = config.get("models", {}).get("bpr_factors", 50)
        models.append(BPRRecommender(n_factors=n_factors))
    
    return models


def train_and_evaluate_models(models: List, train_df: pd.DataFrame, 
                            val_df: pd.DataFrame, test_df: pd.DataFrame,
                            config: Dict) -> ModelLeaderboard:
    """Train and evaluate all models.
    
    Args:
        models: List of model instances.
        train_df: Training data.
        val_df: Validation data.
        test_df: Test data.
        config: Configuration dictionary.
        
    Returns:
        Model leaderboard with results.
    """
    leaderboard = ModelLeaderboard()
    metrics_calculator = RecommendationMetrics()
    
    k_values = config.get("evaluation", {}).get("k_values", [5, 10, 20])
    
    for model in models:
        logger.info(f"Training {model.name}...")
        
        try:
            # Train model
            start_time = time.time()
            model.fit(train_df)
            training_time = time.time() - start_time
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Evaluate on test set
            logger.info(f"Evaluating {model.name}...")
            start_time = time.time()
            
            # Recommendation metrics
            rec_metrics = metrics_calculator.evaluate_model(model, test_df, k_values)
            
            # Rating prediction metrics (if applicable)
            if hasattr(model, 'predict'):
                pred_metrics = metrics_calculator.evaluate_rating_prediction(model, test_df)
                rec_metrics.update(pred_metrics)
            
            evaluation_time = time.time() - start_time
            logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
            
            # Add timing information
            rec_metrics["training_time"] = training_time
            rec_metrics["evaluation_time"] = evaluation_time
            
            # Add to leaderboard
            leaderboard.add_model(model.name, rec_metrics)
            
            logger.info(f"{model.name} - NDCG@10: {rec_metrics.get('NDCG@10', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Failed to train/evaluate {model.name}: {e}")
            continue
    
    return leaderboard


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train and evaluate matrix factorization models")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing data files")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    data_loader = DataLoader(args.data_dir)
    
    interactions_df = data_loader.load_interactions()
    items_df = data_loader.load_items()
    users_df = data_loader.load_users()
    
    logger.info(f"Loaded {len(interactions_df)} interactions, {len(items_df)} items, {len(users_df)} users")
    
    # Split data
    logger.info("Splitting data...")
    splitter = DataSplitter(
        test_size=config.get("data", {}).get("test_size", 0.2),
        val_size=config.get("data", {}).get("val_size", 0.1),
        random_state=args.seed
    )
    
    split_method = config.get("data", {}).get("split_method", "random")
    if split_method == "temporal":
        train_df, val_df, test_df = splitter.temporal_split(interactions_df)
    elif split_method == "user_based":
        min_interactions = config.get("data", {}).get("min_interactions", 5)
        train_df, val_df, test_df = splitter.user_based_split(interactions_df, min_interactions)
    else:
        train_df, val_df, test_df = splitter.random_split(interactions_df)
    
    logger.info(f"Data split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    # Create models
    logger.info("Creating models...")
    models = create_models(config)
    logger.info(f"Created {len(models)} models")
    
    # Train and evaluate models
    logger.info("Starting training and evaluation...")
    leaderboard = train_and_evaluate_models(models, train_df, val_df, test_df, config)
    
    # Save results
    logger.info("Saving results...")
    
    # Save leaderboard
    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard.save_leaderboard(str(leaderboard_path))
    
    # Print leaderboard
    leaderboard.print_leaderboard()
    
    # Save detailed results
    results_path = output_dir / "detailed_results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(leaderboard.results, f, default_flow_style=False)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info("Training and evaluation completed!")


if __name__ == "__main__":
    main()
