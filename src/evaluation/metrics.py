"""Evaluation metrics for recommendation systems."""

import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


class RecommendationMetrics:
    """Collection of recommendation evaluation metrics."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def precision_at_k(self, recommended_items: List[Union[int, str]], 
                      relevant_items: Set[Union[int, str]], k: int) -> float:
        """Calculate Precision@K.
        
        Args:
            recommended_items: List of recommended items.
            relevant_items: Set of relevant (ground truth) items.
            k: Number of top recommendations to consider.
            
        Returns:
            Precision@K score.
        """
        if k == 0:
            return 0.0
            
        top_k_recommendations = recommended_items[:k]
        if not top_k_recommendations:
            return 0.0
            
        relevant_recommended = sum(1 for item in top_k_recommendations if item in relevant_items)
        return relevant_recommended / len(top_k_recommendations)
    
    def recall_at_k(self, recommended_items: List[Union[int, str]], 
                   relevant_items: Set[Union[int, str]], k: int) -> float:
        """Calculate Recall@K.
        
        Args:
            recommended_items: List of recommended items.
            relevant_items: Set of relevant (ground truth) items.
            k: Number of top recommendations to consider.
            
        Returns:
            Recall@K score.
        """
        if not relevant_items:
            return 0.0
            
        top_k_recommendations = recommended_items[:k]
        if not top_k_recommendations:
            return 0.0
            
        relevant_recommended = sum(1 for item in top_k_recommendations if item in relevant_items)
        return relevant_recommended / len(relevant_items)
    
    def hit_rate_at_k(self, recommended_items: List[Union[int, str]], 
                     relevant_items: Set[Union[int, str]], k: int) -> float:
        """Calculate Hit Rate@K.
        
        Args:
            recommended_items: List of recommended items.
            relevant_items: Set of relevant (ground truth) items.
            k: Number of top recommendations to consider.
            
        Returns:
            Hit Rate@K score (1 if any relevant item in top-k, 0 otherwise).
        """
        top_k_recommendations = recommended_items[:k]
        return 1.0 if any(item in relevant_items for item in top_k_recommendations) else 0.0
    
    def ndcg_at_k(self, recommended_items: List[Union[int, str]], 
                  relevant_items: Set[Union[int, str]], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            recommended_items: List of recommended items.
            relevant_items: Set of relevant (ground truth) items.
            k: Number of top recommendations to consider.
            
        Returns:
            NDCG@K score.
        """
        if not relevant_items:
            return 0.0
            
        top_k_recommendations = recommended_items[:k]
        if not top_k_recommendations:
            return 0.0
            
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_recommendations):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
                
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(relevant_items), k)):
            idcg += 1.0 / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0
    
    def map_at_k(self, recommended_items: List[Union[int, str]], 
                 relevant_items: Set[Union[int, str]], k: int) -> float:
        """Calculate Mean Average Precision@K.
        
        Args:
            recommended_items: List of recommended items.
            relevant_items: Set of relevant (ground truth) items.
            k: Number of top recommendations to consider.
            
        Returns:
            MAP@K score.
        """
        if not relevant_items:
            return 0.0
            
        top_k_recommendations = recommended_items[:k]
        if not top_k_recommendations:
            return 0.0
            
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recommendations):
            if item in relevant_items:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
                
        return precision_sum / len(relevant_items) if relevant_items else 0.0
    
    def coverage(self, all_items: Set[Union[int, str]], 
                recommended_items_per_user: List[List[Union[int, str]]]) -> float:
        """Calculate catalog coverage.
        
        Args:
            all_items: Set of all items in the catalog.
            recommended_items_per_user: List of recommendations for each user.
            
        Returns:
            Coverage score (fraction of items recommended).
        """
        if not all_items:
            return 0.0
            
        recommended_items = set()
        for user_recommendations in recommended_items_per_user:
            recommended_items.update(user_recommendations)
            
        return len(recommended_items) / len(all_items)
    
    def diversity(self, recommended_items_per_user: List[List[Union[int, str]]], 
                  item_features: Optional[Dict[Union[int, str], List]] = None) -> float:
        """Calculate diversity of recommendations.
        
        Args:
            recommended_items_per_user: List of recommendations for each user.
            item_features: Optional item features for calculating diversity.
            
        Returns:
            Diversity score (average pairwise dissimilarity).
        """
        if not recommended_items_per_user:
            return 0.0
            
        total_diversity = 0.0
        count = 0
        
        for user_recommendations in recommended_items_per_user:
            if len(user_recommendations) < 2:
                continue
                
            user_diversity = 0.0
            pair_count = 0
            
            for i in range(len(user_recommendations)):
                for j in range(i + 1, len(user_recommendations)):
                    item1, item2 = user_recommendations[i], user_recommendations[j]
                    
                    if item_features and item1 in item_features and item2 in item_features:
                        # Calculate cosine dissimilarity
                        features1 = np.array(item_features[item1])
                        features2 = np.array(item_features[item2])
                        
                        if np.linalg.norm(features1) > 0 and np.linalg.norm(features2) > 0:
                            similarity = np.dot(features1, features2) / (
                                np.linalg.norm(features1) * np.linalg.norm(features2)
                            )
                            dissimilarity = 1 - similarity
                        else:
                            dissimilarity = 1.0
                    else:
                        # Simple dissimilarity: different items are dissimilar
                        dissimilarity = 1.0 if item1 != item2 else 0.0
                        
                    user_diversity += dissimilarity
                    pair_count += 1
                    
            if pair_count > 0:
                total_diversity += user_diversity / pair_count
                count += 1
                
        return total_diversity / count if count > 0 else 0.0
    
    def novelty(self, recommended_items_per_user: List[List[Union[int, str]]], 
                item_popularity: Dict[Union[int, str], float]) -> float:
        """Calculate novelty of recommendations.
        
        Args:
            recommended_items_per_user: List of recommendations for each user.
            item_popularity: Dictionary mapping items to their popularity scores.
            
        Returns:
            Novelty score (average negative log popularity).
        """
        if not recommended_items_per_user:
            return 0.0
            
        total_novelty = 0.0
        total_items = 0
        
        for user_recommendations in recommended_items_per_user:
            for item in user_recommendations:
                popularity = item_popularity.get(item, 0.0)
                if popularity > 0:
                    novelty = -np.log2(popularity)
                else:
                    novelty = 0.0  # Handle zero popularity
                total_novelty += novelty
                total_items += 1
                
        return total_novelty / total_items if total_items > 0 else 0.0
    
    def evaluate_model(self, model, test_df: pd.DataFrame, 
                      k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """Evaluate a recommendation model on test data.
        
        Args:
            model: Trained recommendation model.
            test_df: Test DataFrame with user-item interactions.
            k_values: List of k values for evaluation.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {}
        
        # Group test data by user
        test_by_user = test_df.groupby("user_id")["item_id"].apply(set).to_dict()
        
        # Generate recommendations for each user
        all_recommendations = []
        precision_scores = {k: [] for k in k_values}
        recall_scores = {k: [] for k in k_values}
        hit_rate_scores = {k: [] for k in k_values}
        ndcg_scores = {k: [] for k in k_values}
        map_scores = {k: [] for k in k_values}
        
        for user_id, relevant_items in test_by_user.items():
            try:
                recommendations = model.recommend(user_id, n_recommendations=max(k_values))
                recommended_items = [item for item, score in recommendations]
                all_recommendations.append(recommended_items)
                
                for k in k_values:
                    precision_scores[k].append(
                        self.precision_at_k(recommended_items, relevant_items, k)
                    )
                    recall_scores[k].append(
                        self.recall_at_k(recommended_items, relevant_items, k)
                    )
                    hit_rate_scores[k].append(
                        self.hit_rate_at_k(recommended_items, relevant_items, k)
                    )
                    ndcg_scores[k].append(
                        self.ndcg_at_k(recommended_items, relevant_items, k)
                    )
                    map_scores[k].append(
                        self.map_at_k(recommended_items, relevant_items, k)
                    )
                    
            except Exception as e:
                logger.warning(f"Failed to generate recommendations for user {user_id}: {e}")
                continue
        
        # Calculate average metrics
        for k in k_values:
            metrics[f"Precision@{k}"] = np.mean(precision_scores[k]) if precision_scores[k] else 0.0
            metrics[f"Recall@{k}"] = np.mean(recall_scores[k]) if recall_scores[k] else 0.0
            metrics[f"HitRate@{k}"] = np.mean(hit_rate_scores[k]) if hit_rate_scores[k] else 0.0
            metrics[f"NDCG@{k}"] = np.mean(ndcg_scores[k]) if ndcg_scores[k] else 0.0
            metrics[f"MAP@{k}"] = np.mean(map_scores[k]) if map_scores[k] else 0.0
        
        # Calculate coverage
        all_items = set(test_df["item_id"].unique())
        metrics["Coverage"] = self.coverage(all_items, all_recommendations)
        
        return metrics
    
    def evaluate_rating_prediction(self, model, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate rating prediction accuracy.
        
        Args:
            model: Trained recommendation model.
            test_df: Test DataFrame with user-item interactions and ratings.
            
        Returns:
            Dictionary of rating prediction metrics.
        """
        predictions = []
        actual_ratings = []
        
        for _, row in test_df.iterrows():
            try:
                pred_rating = model.predict(row["user_id"], row["item_id"])
                predictions.append(pred_rating)
                actual_ratings.append(row["rating"])
            except Exception as e:
                logger.warning(f"Failed to predict rating for user {row['user_id']}, item {row['item_id']}: {e}")
                continue
        
        if not predictions:
            return {"RMSE": float('inf'), "MAE": float('inf')}
        
        predictions = np.array(predictions)
        actual_ratings = np.array(actual_ratings)
        
        rmse = np.sqrt(mean_squared_error(actual_ratings, predictions))
        mae = mean_absolute_error(actual_ratings, predictions)
        
        return {"RMSE": rmse, "MAE": mae}


class ModelLeaderboard:
    """Leaderboard for comparing recommendation models."""
    
    def __init__(self):
        """Initialize the leaderboard."""
        self.results: List[Dict] = []
        
    def add_model(self, model_name: str, metrics: Dict[str, float], 
                  config: Optional[Dict] = None) -> None:
        """Add model results to the leaderboard.
        
        Args:
            model_name: Name of the model.
            metrics: Dictionary of evaluation metrics.
            config: Optional model configuration.
        """
        result = {
            "model": model_name,
            "metrics": metrics,
            "config": config or {}
        }
        self.results.append(result)
        
    def get_leaderboard(self, metric: str = "NDCG@10", ascending: bool = False) -> pd.DataFrame:
        """Get sorted leaderboard for a specific metric.
        
        Args:
            metric: Metric to sort by.
            ascending: Whether to sort in ascending order.
            
        Returns:
            DataFrame with sorted results.
        """
        if not self.results:
            return pd.DataFrame()
            
        leaderboard_data = []
        for result in self.results:
            row = {"model": result["model"]}
            row.update(result["metrics"])
            leaderboard_data.append(row)
            
        df = pd.DataFrame(leaderboard_data)
        
        if metric in df.columns:
            df = df.sort_values(metric, ascending=ascending)
            
        return df
    
    def print_leaderboard(self, metric: str = "NDCG@10", top_k: int = 10) -> None:
        """Print formatted leaderboard.
        
        Args:
            metric: Metric to sort by.
            top_k: Number of top models to show.
        """
        leaderboard = self.get_leaderboard(metric)
        
        if leaderboard.empty:
            print("No results in leaderboard.")
            return
            
        print(f"\n{'='*60}")
        print(f"LEADERBOARD - Sorted by {metric}")
        print(f"{'='*60}")
        
        # Show top-k results
        top_results = leaderboard.head(top_k)
        
        # Format the output
        for idx, (_, row) in enumerate(top_results.iterrows(), 1):
            print(f"{idx:2d}. {row['model']:<20} {metric}: {row[metric]:.4f}")
            
        print(f"{'='*60}")
        
    def save_leaderboard(self, filepath: str, metric: str = "NDCG@10") -> None:
        """Save leaderboard to CSV file.
        
        Args:
            filepath: Path to save the CSV file.
            metric: Metric to sort by.
        """
        leaderboard = self.get_leaderboard(metric)
        leaderboard.to_csv(filepath, index=False)
        logger.info(f"Leaderboard saved to {filepath}")
