#!/usr/bin/env python3
"""Simple demo script for matrix factorization recommendations."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
from data.loader import DataLoader, DataSplitter
from models.matrix_factorization import SVDRecommender, NMFRecommender, PopularityRecommender
from evaluation.metrics import RecommendationMetrics


def main():
    """Run a simple demo of the recommendation system."""
    print("🎯 Matrix Factorization Recommendations Demo")
    print("=" * 50)
    
    # Load data
    print("\n📊 Loading data...")
    data_loader = DataLoader("data")
    interactions_df = data_loader.load_interactions()
    items_df = data_loader.load_items()
    
    print(f"Loaded {len(interactions_df)} interactions and {len(items_df)} items")
    
    # Split data
    print("\n✂️  Splitting data...")
    splitter = DataSplitter(random_state=42)
    train_df, val_df, test_df = splitter.random_split(interactions_df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Train models
    print("\n🤖 Training models...")
    models = {
        "Popularity": PopularityRecommender(),
        "SVD": SVDRecommender(n_factors=20),
        "NMF": NMFRecommender(n_factors=20)
    }
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(train_df)
    
    # Evaluate models
    print("\n📈 Evaluating models...")
    metrics_calculator = RecommendationMetrics()
    
    for name, model in models.items():
        metrics = metrics_calculator.evaluate_model(model, test_df, k_values=[5, 10])
        print(f"  {name}:")
        print(f"    Precision@10: {metrics['Precision@10']:.4f}")
        print(f"    Recall@10: {metrics['Recall@10']:.4f}")
        print(f"    NDCG@10: {metrics['NDCG@10']:.4f}")
        print(f"    Hit Rate@10: {metrics['HitRate@10']:.4f}")
    
    # Generate recommendations for a sample user
    print("\n🎯 Generating recommendations...")
    sample_user = test_df['user_id'].iloc[0]
    print(f"Recommendations for user {sample_user}:")
    
    for name, model in models.items():
        print(f"\n  {name} recommendations:")
        recommendations = model.recommend(sample_user, n_recommendations=5)
        
        for i, (item_id, score) in enumerate(recommendations, 1):
            item_info = items_df[items_df['item_id'] == item_id]
            if not item_info.empty:
                title = item_info.iloc[0].get('title', f'Item {item_id}')
                category = item_info.iloc[0].get('category', 'Unknown')
                print(f"    {i}. {title} ({category}) - Score: {score:.3f}")
            else:
                print(f"    {i}. Item {item_id} - Score: {score:.3f}")
    
    # Show user's actual ratings
    print(f"\n📝 User {sample_user}'s actual ratings:")
    user_ratings = test_df[test_df['user_id'] == sample_user].head(5)
    for _, rating in user_ratings.iterrows():
        item_info = items_df[items_df['item_id'] == rating['item_id']]
        if not item_info.empty:
            title = item_info.iloc[0].get('title', f'Item {rating["item_id"]}')
            print(f"  {title}: {rating['rating']} stars")
    
    print("\n✅ Demo completed!")
    print("\nTo run the full training and evaluation:")
    print("  python3 scripts/train_evaluate.py --config configs/default.yaml")
    print("\nTo launch the interactive demo:")
    print("  streamlit run demo.py")


if __name__ == "__main__":
    main()
