"""Unit tests for matrix factorization recommendation system."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.data.loader import DataLoader, DataSplitter, create_user_item_matrix
from src.models.matrix_factorization import (
    PopularityRecommender, UserKNNRecommender, ItemKNNRecommender,
    SVDRecommender, NMFRecommender, BaseRecommender
)
from src.evaluation.metrics import RecommendationMetrics, ModelLeaderboard


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader("test_data")
        assert loader.data_dir.name == "test_data"
        assert loader.interactions_df is None
        assert loader.items_df is None
        assert loader.users_df is None
    
    def test_create_synthetic_interactions(self):
        """Test synthetic interactions creation."""
        loader = DataLoader("test_data")
        df = loader._create_synthetic_interactions()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "user_id" in df.columns
        assert "item_id" in df.columns
        assert "rating" in df.columns
        assert "timestamp" in df.columns
        assert df["rating"].min() >= 1
        assert df["rating"].max() <= 5
    
    def test_create_synthetic_items(self):
        """Test synthetic items creation."""
        loader = DataLoader("test_data")
        df = loader._create_synthetic_items()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "item_id" in df.columns
        assert "title" in df.columns
        assert "category" in df.columns
    
    def test_create_synthetic_users(self):
        """Test synthetic users creation."""
        loader = DataLoader("test_data")
        df = loader._create_synthetic_users()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "user_id" in df.columns
        assert "age_group" in df.columns
        assert "gender" in df.columns


class TestDataSplitter:
    """Test cases for DataSplitter class."""
    
    def test_init(self):
        """Test DataSplitter initialization."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)
        assert splitter.test_size == 0.2
        assert splitter.val_size == 0.1
        assert splitter.random_state == 42
    
    def test_random_split(self):
        """Test random data splitting."""
        splitter = DataSplitter(random_state=42)
        
        # Create test data
        df = pd.DataFrame({
            "user_id": [0, 0, 1, 1, 2, 2],
            "item_id": [0, 1, 0, 2, 1, 2],
            "rating": [5, 4, 3, 2, 4, 5],
            "timestamp": [1, 2, 3, 4, 5, 6]
        })
        
        train_df, val_df, test_df = splitter.random_split(df)
        
        assert len(train_df) + len(val_df) + len(test_df) == len(df)
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
    
    def test_temporal_split(self):
        """Test temporal data splitting."""
        splitter = DataSplitter(random_state=42)
        
        # Create test data
        df = pd.DataFrame({
            "user_id": [0, 0, 1, 1, 2, 2],
            "item_id": [0, 1, 0, 2, 1, 2],
            "rating": [5, 4, 3, 2, 4, 5],
            "timestamp": [1, 2, 3, 4, 5, 6]
        })
        
        train_df, val_df, test_df = splitter.temporal_split(df)
        
        assert len(train_df) + len(val_df) + len(test_df) == len(df)
        assert train_df["timestamp"].max() <= val_df["timestamp"].min()
        assert val_df["timestamp"].max() <= test_df["timestamp"].min()


class TestCreateUserItemMatrix:
    """Test cases for create_user_item_matrix function."""
    
    def test_create_matrix(self):
        """Test user-item matrix creation."""
        df = pd.DataFrame({
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 0, 2],
            "rating": [5, 4, 3, 2]
        })
        
        matrix, user_id_to_idx, item_id_to_idx = create_user_item_matrix(df)
        
        assert matrix.shape == (2, 3)  # 2 users, 3 items
        assert matrix[0, 0] == 5  # User 0, Item 0
        assert matrix[0, 1] == 4  # User 0, Item 1
        assert matrix[1, 0] == 3  # User 1, Item 0
        assert matrix[1, 2] == 2  # User 1, Item 2
        
        assert user_id_to_idx[0] == 0
        assert user_id_to_idx[1] == 1
        assert item_id_to_idx[0] == 0
        assert item_id_to_idx[1] == 1
        assert item_id_to_idx[2] == 2


class TestBaseRecommender:
    """Test cases for BaseRecommender abstract class."""
    
    def test_init(self):
        """Test BaseRecommender initialization."""
        # Create a concrete implementation for testing
        class TestRecommender(BaseRecommender):
            def fit(self, interactions_df):
                self.is_fitted = True
            
            def predict(self, user_id, item_id):
                return 3.0
            
            def recommend(self, user_id, n_recommendations=10, exclude_rated=True):
                return [("item1", 3.0), ("item2", 2.5)]
        
        model = TestRecommender("Test")
        assert model.name == "Test"
        assert not model.is_fitted


class TestPopularityRecommender:
    """Test cases for PopularityRecommender class."""
    
    def test_fit_and_predict(self):
        """Test PopularityRecommender fit and predict."""
        model = PopularityRecommender()
        
        # Create test data
        df = pd.DataFrame({
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 0, 2],
            "rating": [5, 4, 3, 2]
        })
        
        model.fit(df)
        assert model.is_fitted
        
        # Test prediction
        prediction = model.predict(0, 0)
        assert prediction == 4.0  # Average of ratings 5 and 3 for item 0
    
    def test_recommend(self):
        """Test PopularityRecommender recommendations."""
        model = PopularityRecommender()
        
        df = pd.DataFrame({
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 0, 2],
            "rating": [5, 4, 3, 2]
        })
        
        model.fit(df)
        recommendations = model.recommend(0, n_recommendations=2)
        
        assert len(recommendations) <= 2
        assert all(isinstance(item, tuple) for item in recommendations)
        assert all(len(item) == 2 for item in recommendations)


class TestSVDRecommender:
    """Test cases for SVDRecommender class."""
    
    def test_fit_and_predict(self):
        """Test SVDRecommender fit and predict."""
        model = SVDRecommender(n_factors=2)
        
        # Create test data
        df = pd.DataFrame({
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 0, 2],
            "rating": [5, 4, 3, 2]
        })
        
        model.fit(df)
        assert model.is_fitted
        
        # Test prediction
        prediction = model.predict(0, 0)
        assert isinstance(prediction, float)
        assert 0 <= prediction <= 5
    
    def test_recommend(self):
        """Test SVDRecommender recommendations."""
        model = SVDRecommender(n_factors=2)
        
        df = pd.DataFrame({
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 0, 2],
            "rating": [5, 4, 3, 2]
        })
        
        model.fit(df)
        recommendations = model.recommend(0, n_recommendations=2)
        
        assert len(recommendations) <= 2
        assert all(isinstance(item, tuple) for item in recommendations)


class TestNMFRecommender:
    """Test cases for NMFRecommender class."""
    
    def test_fit_and_predict(self):
        """Test NMFRecommender fit and predict."""
        model = NMFRecommender(n_factors=2)
        
        # Create test data
        df = pd.DataFrame({
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 0, 2],
            "rating": [5, 4, 3, 2]
        })
        
        model.fit(df)
        assert model.is_fitted
        
        # Test prediction
        prediction = model.predict(0, 0)
        assert isinstance(prediction, float)
        assert 0 <= prediction <= 5


class TestRecommendationMetrics:
    """Test cases for RecommendationMetrics class."""
    
    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        metrics = RecommendationMetrics()
        
        recommended_items = ["item1", "item2", "item3", "item4", "item5"]
        relevant_items = {"item1", "item3", "item5"}
        
        precision = metrics.precision_at_k(recommended_items, relevant_items, k=5)
        assert precision == 0.6  # 3 relevant out of 5 recommended
    
    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        metrics = RecommendationMetrics()
        
        recommended_items = ["item1", "item2", "item3", "item4", "item5"]
        relevant_items = {"item1", "item3", "item5", "item6"}
        
        recall = metrics.recall_at_k(recommended_items, relevant_items, k=5)
        assert recall == 0.75  # 3 relevant out of 4 total relevant
    
    def test_hit_rate_at_k(self):
        """Test Hit Rate@K calculation."""
        metrics = RecommendationMetrics()
        
        recommended_items = ["item1", "item2", "item3"]
        relevant_items = {"item2", "item4"}
        
        hit_rate = metrics.hit_rate_at_k(recommended_items, relevant_items, k=3)
        assert hit_rate == 1.0  # item2 is in recommendations
        
        hit_rate = metrics.hit_rate_at_k(recommended_items, {"item4"}, k=3)
        assert hit_rate == 0.0  # item4 is not in recommendations
    
    def test_ndcg_at_k(self):
        """Test NDCG@K calculation."""
        metrics = RecommendationMetrics()
        
        recommended_items = ["item1", "item2", "item3"]
        relevant_items = {"item1", "item3"}
        
        ndcg = metrics.ndcg_at_k(recommended_items, relevant_items, k=3)
        assert 0 <= ndcg <= 1
    
    def test_coverage(self):
        """Test coverage calculation."""
        metrics = RecommendationMetrics()
        
        all_items = {"item1", "item2", "item3", "item4", "item5"}
        recommended_items_per_user = [
            ["item1", "item2"],
            ["item2", "item3"],
            ["item1", "item4"]
        ]
        
        coverage = metrics.coverage(all_items, recommended_items_per_user)
        assert coverage == 0.8  # 4 out of 5 items recommended
    
    def test_diversity(self):
        """Test diversity calculation."""
        metrics = RecommendationMetrics()
        
        recommended_items_per_user = [
            ["item1", "item2", "item3"],
            ["item1", "item2", "item3"]
        ]
        
        diversity = metrics.diversity(recommended_items_per_user)
        assert 0 <= diversity <= 1


class TestModelLeaderboard:
    """Test cases for ModelLeaderboard class."""
    
    def test_init(self):
        """Test ModelLeaderboard initialization."""
        leaderboard = ModelLeaderboard()
        assert leaderboard.results == []
    
    def test_add_model(self):
        """Test adding model to leaderboard."""
        leaderboard = ModelLeaderboard()
        
        metrics = {"NDCG@10": 0.5, "Precision@10": 0.3}
        config = {"n_factors": 50}
        
        leaderboard.add_model("TestModel", metrics, config)
        
        assert len(leaderboard.results) == 1
        assert leaderboard.results[0]["model"] == "TestModel"
        assert leaderboard.results[0]["metrics"] == metrics
        assert leaderboard.results[0]["config"] == config
    
    def test_get_leaderboard(self):
        """Test getting sorted leaderboard."""
        leaderboard = ModelLeaderboard()
        
        # Add multiple models
        leaderboard.add_model("Model1", {"NDCG@10": 0.5})
        leaderboard.add_model("Model2", {"NDCG@10": 0.7})
        leaderboard.add_model("Model3", {"NDCG@10": 0.3})
        
        result_df = leaderboard.get_leaderboard("NDCG@10")
        
        assert len(result_df) == 3
        assert result_df.iloc[0]["model"] == "Model2"  # Highest NDCG@10
        assert result_df.iloc[2]["model"] == "Model3"  # Lowest NDCG@10


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete training and evaluation pipeline."""
        # Create test data
        df = pd.DataFrame({
            "user_id": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "item_id": [0, 1, 2, 0, 1, 3, 1, 2, 3],
            "rating": [5, 4, 3, 4, 5, 2, 3, 4, 5],
            "timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 9]
        })
        
        # Split data
        splitter = DataSplitter(random_state=42)
        train_df, val_df, test_df = splitter.random_split(df)
        
        # Train model
        model = SVDRecommender(n_factors=2)
        model.fit(train_df)
        
        # Evaluate
        metrics_calculator = RecommendationMetrics()
        metrics = metrics_calculator.evaluate_model(model, test_df, k_values=[5, 10])
        
        # Check that metrics were calculated
        assert "Precision@5" in metrics
        assert "Recall@5" in metrics
        assert "NDCG@5" in metrics
        assert all(0 <= v <= 1 for v in metrics.values() if isinstance(v, (int, float)))


if __name__ == "__main__":
    pytest.main([__file__])
