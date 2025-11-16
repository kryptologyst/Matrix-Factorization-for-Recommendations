"""Streamlit demo for matrix factorization recommendations."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.loader import DataLoader
from src.models.matrix_factorization import (
    PopularityRecommender, UserKNNRecommender, ItemKNNRecommender,
    SVDRecommender, NMFRecommender, ALSRecommender, BPRRecommender
)
from src.evaluation.metrics import RecommendationMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Matrix Factorization Recommendations",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-item {
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and cache data."""
    data_loader = DataLoader(data_dir)
    interactions_df = data_loader.load_interactions()
    items_df = data_loader.load_items()
    users_df = data_loader.load_users()
    return interactions_df, items_df, users_df


@st.cache_resource
def load_trained_models(models_dir: str = "models") -> Dict[str, any]:
    """Load trained models from disk."""
    models = {}
    models_path = Path(models_dir)
    
    if not models_path.exists():
        st.warning("No trained models found. Please run training first.")
        return models
    
    for model_file in models_path.glob("*.pkl"):
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                models[model.name] = model
        except Exception as e:
            st.error(f"Failed to load model {model_file}: {e}")
    
    return models


def train_models_on_demand(interactions_df: pd.DataFrame) -> Dict[str, any]:
    """Train models on demand if not available."""
    st.info("Training models on demand...")
    
    models = {}
    
    # Create a subset of models for demo
    model_configs = [
        ("Popularity", PopularityRecommender()),
        ("UserKNN", UserKNNRecommender(k=20)),
        ("ItemKNN", ItemKNNRecommender(k=20)),
        ("SVD", SVDRecommender(n_factors=20)),
        ("NMF", NMFRecommender(n_factors=20))
    ]
    
    progress_bar = st.progress(0)
    
    for i, (name, model) in enumerate(model_configs):
        try:
            with st.spinner(f"Training {name}..."):
                model.fit(interactions_df)
                models[name] = model
            progress_bar.progress((i + 1) / len(model_configs))
        except Exception as e:
            st.error(f"Failed to train {name}: {e}")
    
    return models


def display_user_recommendations(user_id: str, models: Dict[str, any], 
                                items_df: pd.DataFrame, n_recommendations: int = 10):
    """Display recommendations for a specific user."""
    st.subheader(f"Recommendations for User {user_id}")
    
    # Create columns for different models
    cols = st.columns(len(models))
    
    for i, (model_name, model) in enumerate(models.items()):
        with cols[i]:
            st.write(f"**{model_name}**")
            
            try:
                recommendations = model.recommend(user_id, n_recommendations=n_recommendations)
                
                if not recommendations:
                    st.write("No recommendations available")
                    continue
                
                for j, (item_id, score) in enumerate(recommendations[:5]):  # Show top 5
                    # Get item details
                    item_info = items_df[items_df["item_id"] == item_id]
                    if not item_info.empty:
                        title = item_info.iloc[0].get("title", f"Item {item_id}")
                        category = item_info.iloc[0].get("category", "Unknown")
                    else:
                        title = f"Item {item_id}"
                        category = "Unknown"
                    
                    st.markdown(f"""
                    <div class="recommendation-item">
                        <strong>{j+1}. {title}</strong><br>
                        <small>Category: {category}</small><br>
                        <small>Score: {score:.3f}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")


def display_item_similarity(item_id: str, models: Dict[str, any], 
                           items_df: pd.DataFrame, n_similar: int = 10):
    """Display similar items for a specific item."""
    st.subheader(f"Items Similar to Item {item_id}")
    
    # Get item details
    item_info = items_df[items_df["item_id"] == item_id]
    if not item_info.empty:
        title = item_info.iloc[0].get("title", f"Item {item_id}")
        category = item_info.iloc[0].get("category", "Unknown")
        st.write(f"**{title}** ({category})")
    
    # For demonstration, we'll use item-based collaborative filtering
    if "ItemKNN" in models:
        model = models["ItemKNN"]
        
        # Find similar items (this is a simplified approach)
        try:
            # Get all items
            all_items = items_df["item_id"].unique()
            
            # Calculate similarity scores (simplified)
            similarities = []
            for other_item in all_items:
                if other_item != item_id:
                    # This is a simplified similarity calculation
                    # In practice, you'd use the model's similarity matrix
                    similarities.append((other_item, 0.5))  # Placeholder
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Display similar items
            for i, (similar_item_id, similarity) in enumerate(similarities[:n_similar]):
                similar_item_info = items_df[items_df["item_id"] == similar_item_id]
                if not similar_item_info.empty:
                    similar_title = similar_item_info.iloc[0].get("title", f"Item {similar_item_id}")
                    similar_category = similar_item_info.iloc[0].get("category", "Unknown")
                    
                    st.markdown(f"""
                    <div class="recommendation-item">
                        <strong>{i+1}. {similar_title}</strong><br>
                        <small>Category: {similar_category}</small><br>
                        <small>Similarity: {similarity:.3f}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error finding similar items: {e}")


def display_model_comparison(models: Dict[str, any], test_df: pd.DataFrame):
    """Display model comparison metrics."""
    st.subheader("Model Performance Comparison")
    
    metrics_calculator = RecommendationMetrics()
    k_values = [5, 10, 20]
    
    # Calculate metrics for each model
    model_metrics = {}
    
    progress_bar = st.progress(0)
    
    for i, (model_name, model) in enumerate(models.items()):
        try:
            with st.spinner(f"Evaluating {model_name}..."):
                metrics = metrics_calculator.evaluate_model(model, test_df, k_values)
                model_metrics[model_name] = metrics
            progress_bar.progress((i + 1) / len(models))
        except Exception as e:
            st.error(f"Failed to evaluate {model_name}: {e}")
    
    if not model_metrics:
        st.warning("No metrics available for comparison")
        return
    
    # Create comparison plots
    metrics_to_plot = ["Precision@10", "Recall@10", "NDCG@10", "HitRate@10"]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metrics_to_plot,
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    for i, metric in enumerate(metrics_to_plot):
        row = i // 2 + 1
        col = i % 2 + 1
        
        model_names = list(model_metrics.keys())
        metric_values = [model_metrics[model].get(metric, 0) for model in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=metric_values, name=metric),
            row=row, col=col
        )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Model Performance Comparison"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display metrics table
    st.subheader("Detailed Metrics")
    
    metrics_df = pd.DataFrame(model_metrics).T
    st.dataframe(metrics_df, use_container_width=True)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Matrix Factorization Recommendations</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["User Recommendations", "Item Similarity", "Model Comparison", "Data Overview"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        interactions_df, items_df, users_df = load_data()
    
    # Load or train models
    models = load_trained_models()
    
    if not models:
        st.info("No pre-trained models found. Training models on demand...")
        models = train_models_on_demand(interactions_df)
    
    if not models:
        st.error("Failed to load or train any models. Please check the configuration.")
        return
    
    st.sidebar.success(f"Loaded {len(models)} models")
    
    # Main content based on selected page
    if page == "User Recommendations":
        st.header("User Recommendations")
        
        # User selection
        available_users = interactions_df["user_id"].unique()
        selected_user = st.selectbox("Select a user", available_users)
        
        # Number of recommendations
        n_recommendations = st.slider("Number of recommendations", 5, 20, 10)
        
        # Display recommendations
        if st.button("Generate Recommendations"):
            display_user_recommendations(selected_user, models, items_df, n_recommendations)
    
    elif page == "Item Similarity":
        st.header("Item Similarity")
        
        # Item selection
        available_items = items_df["item_id"].unique()
        selected_item = st.selectbox("Select an item", available_items)
        
        # Number of similar items
        n_similar = st.slider("Number of similar items", 5, 20, 10)
        
        # Display similar items
        if st.button("Find Similar Items"):
            display_item_similarity(selected_item, models, items_df, n_similar)
    
    elif page == "Model Comparison":
        st.header("Model Comparison")
        
        # Use a subset of test data for faster evaluation
        test_df = interactions_df.sample(min(1000, len(interactions_df)), random_state=42)
        
        if st.button("Compare Models"):
            display_model_comparison(models, test_df)
    
    elif page == "Data Overview":
        st.header("Data Overview")
        
        # Data statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Users", len(users_df))
            st.metric("Total Items", len(items_df))
        
        with col2:
            st.metric("Total Interactions", len(interactions_df))
            st.metric("Avg Rating", f"{interactions_df['rating'].mean():.2f}")
        
        with col3:
            st.metric("Sparsity", f"{1 - len(interactions_df) / (len(users_df) * len(items_df)):.2%}")
            st.metric("Unique Users", interactions_df["user_id"].nunique())
        
        # Rating distribution
        st.subheader("Rating Distribution")
        rating_counts = interactions_df["rating"].value_counts().sort_index()
        
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title="Rating Distribution",
            labels={"x": "Rating", "y": "Count"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Category distribution
        if "category" in items_df.columns:
            st.subheader("Item Category Distribution")
            category_counts = items_df["category"].value_counts()
            
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Item Categories"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample data
        st.subheader("Sample Data")
        
        st.write("**Interactions:**")
        st.dataframe(interactions_df.head(10), use_container_width=True)
        
        st.write("**Items:**")
        st.dataframe(items_df.head(10), use_container_width=True)
        
        st.write("**Users:**")
        st.dataframe(users_df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
