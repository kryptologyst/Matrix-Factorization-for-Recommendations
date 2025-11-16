"""Data loading and preprocessing utilities for matrix factorization recommendations."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


class DataLoader:
    """Data loader for recommendation system datasets."""
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing the data files.
        """
        self.data_dir = Path(data_dir)
        self.interactions_df: Optional[pd.DataFrame] = None
        self.items_df: Optional[pd.DataFrame] = None
        self.users_df: Optional[pd.DataFrame] = None
        
    def load_interactions(self, file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Load user-item interactions data.
        
        Args:
            file_path: Path to interactions CSV file. If None, uses default path.
            
        Returns:
            DataFrame with columns: user_id, item_id, rating, timestamp
        """
        if file_path is None:
            file_path = self.data_dir / "raw" / "interactions.csv"
            
        if not Path(file_path).exists():
            logger.warning(f"Interactions file not found at {file_path}. Creating synthetic data.")
            return self._create_synthetic_interactions()
            
        df = pd.read_csv(file_path)
        required_cols = ["user_id", "item_id", "rating"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Interactions file must contain columns: {required_cols}")
            
        self.interactions_df = df
        logger.info(f"Loaded {len(df)} interactions from {file_path}")
        return df
    
    def load_items(self, file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Load items metadata.
        
        Args:
            file_path: Path to items CSV file. If None, uses default path.
            
        Returns:
            DataFrame with item metadata.
        """
        if file_path is None:
            file_path = self.data_dir / "raw" / "items.csv"
            
        if not Path(file_path).exists():
            logger.warning(f"Items file not found at {file_path}. Creating synthetic data.")
            return self._create_synthetic_items()
            
        df = pd.read_csv(file_path)
        if "item_id" not in df.columns:
            raise ValueError("Items file must contain 'item_id' column")
            
        self.items_df = df
        logger.info(f"Loaded {len(df)} items from {file_path}")
        return df
    
    def load_users(self, file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Load users metadata.
        
        Args:
            file_path: Path to users CSV file. If None, uses default path.
            
        Returns:
            DataFrame with user metadata.
        """
        if file_path is None:
            file_path = self.data_dir / "raw" / "users.csv"
            
        if not Path(file_path).exists():
            logger.warning(f"Users file not found at {file_path}. Creating synthetic data.")
            return self._create_synthetic_users()
            
        df = pd.read_csv(file_path)
        if "user_id" not in df.columns:
            raise ValueError("Users file must contain 'user_id' column")
            
        self.users_df = df
        logger.info(f"Loaded {len(df)} users from {file_path}")
        return df
    
    def _create_synthetic_interactions(self) -> pd.DataFrame:
        """Create synthetic user-item interactions data."""
        set_seed(42)
        
        n_users = 1000
        n_items = 500
        n_interactions = 10000
        
        # Create user-item pairs with some popularity bias
        user_ids = np.random.choice(range(n_users), n_interactions)
        item_ids = np.random.choice(range(n_items), n_interactions, p=self._get_item_popularity_dist(n_items))
        
        # Create ratings with some patterns
        ratings = np.random.choice([1, 2, 3, 4, 5], n_interactions, p=[0.1, 0.1, 0.2, 0.3, 0.3])
        
        # Add some temporal patterns
        timestamps = np.random.randint(1609459200, 1672531200, n_interactions)  # 2021-2023
        
        df = pd.DataFrame({
            "user_id": user_ids,
            "item_id": item_ids,
            "rating": ratings,
            "timestamp": timestamps
        })
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["user_id", "item_id"])
        
        # Save synthetic data
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        df.to_csv(self.data_dir / "raw" / "interactions.csv", index=False)
        
        logger.info(f"Created synthetic interactions data: {len(df)} interactions")
        return df
    
    def _create_synthetic_items(self) -> pd.DataFrame:
        """Create synthetic items metadata."""
        set_seed(42)
        
        n_items = 500
        categories = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller"]
        
        items = []
        for i in range(n_items):
            items.append({
                "item_id": i,
                "title": f"Item {i}",
                "category": np.random.choice(categories),
                "year": np.random.randint(1990, 2023),
                "rating_avg": np.random.uniform(2.0, 5.0)
            })
        
        df = pd.DataFrame(items)
        
        # Save synthetic data
        df.to_csv(self.data_dir / "raw" / "items.csv", index=False)
        
        logger.info(f"Created synthetic items data: {len(df)} items")
        return df
    
    def _create_synthetic_users(self) -> pd.DataFrame:
        """Create synthetic users metadata."""
        set_seed(42)
        
        n_users = 1000
        age_groups = ["18-25", "26-35", "36-45", "46-55", "55+"]
        genders = ["M", "F", "Other"]
        
        users = []
        for i in range(n_users):
            users.append({
                "user_id": i,
                "age_group": np.random.choice(age_groups),
                "gender": np.random.choice(genders),
                "signup_date": np.random.randint(1609459200, 1672531200)
            })
        
        df = pd.DataFrame(users)
        
        # Save synthetic data
        df.to_csv(self.data_dir / "raw" / "users.csv", index=False)
        
        logger.info(f"Created synthetic users data: {len(df)} users")
        return df
    
    def _get_item_popularity_dist(self, n_items: int) -> np.ndarray:
        """Get popularity distribution for items (power law)."""
        # Power law distribution for item popularity
        popularity = np.power(np.arange(1, n_items + 1), -1.2)
        return popularity / popularity.sum()


class DataSplitter:
    """Utility for splitting recommendation data into train/validation/test sets."""
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        """Initialize the data splitter.
        
        Args:
            test_size: Fraction of data to use for testing.
            val_size: Fraction of data to use for validation.
            random_state: Random seed for reproducibility.
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
    def temporal_split(self, df: pd.DataFrame, time_col: str = "timestamp") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data temporally (chronological split).
        
        Args:
            df: DataFrame with interactions.
            time_col: Name of the timestamp column.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        df_sorted = df.sort_values(time_col)
        
        n_total = len(df_sorted)
        n_test = int(n_total * self.test_size)
        n_val = int(n_total * self.val_size)
        n_train = n_total - n_test - n_val
        
        train_df = df_sorted.iloc[:n_train]
        val_df = df_sorted.iloc[n_train:n_train + n_val]
        test_df = df_sorted.iloc[n_train + n_val:]
        
        logger.info(f"Temporal split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        return train_df, val_df, test_df
    
    def random_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data randomly.
        
        Args:
            df: DataFrame with interactions.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        train_df, temp_df = train_test_split(
            df, test_size=self.test_size + self.val_size, random_state=self.random_state
        )
        
        val_df, test_df = train_test_split(
            temp_df, test_size=self.test_size / (self.test_size + self.val_size), 
            random_state=self.random_state
        )
        
        logger.info(f"Random split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        return train_df, val_df, test_df
    
    def user_based_split(self, df: pd.DataFrame, min_interactions: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by user (leave-last-k per user).
        
        Args:
            df: DataFrame with interactions.
            min_interactions: Minimum interactions per user to include in split.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        train_data = []
        val_data = []
        test_data = []
        
        for user_id in df["user_id"].unique():
            user_interactions = df[df["user_id"] == user_id].sort_values("timestamp")
            
            if len(user_interactions) < min_interactions:
                continue
                
            n_interactions = len(user_interactions)
            n_test = max(1, int(n_interactions * self.test_size))
            n_val = max(1, int(n_interactions * self.val_size))
            n_train = n_interactions - n_test - n_val
            
            if n_train > 0:
                train_data.append(user_interactions.iloc[:n_train])
            if n_val > 0:
                val_data.append(user_interactions.iloc[n_train:n_train + n_val])
            if n_test > 0:
                test_data.append(user_interactions.iloc[n_train + n_val:])
        
        train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
        val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
        test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        
        logger.info(f"User-based split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        return train_df, val_df, test_df


def create_user_item_matrix(df: pd.DataFrame, user_col: str = "user_id", item_col: str = "item_id", 
                          rating_col: str = "rating") -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    """Create user-item interaction matrix.
    
    Args:
        df: DataFrame with interactions.
        user_col: Name of user column.
        item_col: Name of item column.
        rating_col: Name of rating column.
        
    Returns:
        Tuple of (matrix, user_id_to_idx, item_id_to_idx).
    """
    # Create mappings
    unique_users = df[user_col].unique()
    unique_items = df[item_col].unique()
    
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
    
    # Create matrix
    matrix = np.zeros((len(unique_users), len(unique_items)))
    
    for _, row in df.iterrows():
        user_idx = user_id_to_idx[row[user_col]]
        item_idx = item_id_to_idx[row[item_col]]
        matrix[user_idx, item_idx] = row[rating_col]
    
    return matrix, user_id_to_idx, item_id_to_idx
