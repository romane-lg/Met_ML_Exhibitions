"""
Streamlit web application for MET Exhibition Curator.

This app provides an interactive interface for curators to generate
exhibition recommendations based on themes.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data import load_met_data
from src.models import ExhibitionRecommender

# Page configuration
st.set_page_config(
    page_title="MET Exhibition AI Curator",
    page_icon="🎨",
    layout="wide"
)


@st.cache_data
def load_data():
    """Load artwork data."""
    try:
        df = load_met_data("data/raw/met_data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_resource
def load_recommender(df):
    """Load or create recommender system."""
    # For MVP, use simple features
    # TODO: Load actual pre-computed features
    features = np.random.rand(len(df), 50)
    
    recommender = ExhibitionRecommender(
        features=features,
        metadata=df,
        similarity_metric='cosine'
    )
    
    return recommender


def display_artwork(row, show_image=True):
    """Display a single artwork."""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if show_image and 'image_path' in row:
            img_path = Path(row['image_path'])
            if img_path.exists():
                try:
                    image = Image.open(img_path)
                    st.image(image, use_container_width=True)
                except Exception as e:
                    st.write("🖼️ Image unavailable")
            else:
                st.write("🖼️ Image not found")
    
    with col2:
        st.markdown(f"**{row['title']}**")
        if pd.notna(row.get('artist')):
            st.write(f"👤 Artist: {row['artist']}")
        if pd.notna(row.get('department')):
            st.write(f"🏛️ Department: {row['department']}")
        if pd.notna(row.get('objectDate')):
            st.write(f"📅 Date: {row['objectDate']}")
        if pd.notna(row.get('medium')):
            st.write(f"🎨 Medium: {row['medium']}")
        if 'similarity_score' in row:
            st.write(f"⭐ Score: {row['similarity_score']:.3f}")


def main():
    """Main application."""
    
    # Header
    st.title("🎨 MET Exhibition AI Curator")
    st.markdown("*Intelligent artwork recommendations for themed exhibitions*")
    st.divider()
    
    # Load data
    with st.spinner("Loading artwork collection..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        n_exhibitions = st.number_input(
            "Number of Exhibitions",
            min_value=1,
            max_value=10,
            value=3,
            help="How many themed exhibitions to create"
        )
        
        max_pieces = st.slider(
            "Max Pieces per Exhibition",
            min_value=10,
            max_value=50,
            value=25,
            help="Maximum artworks in each exhibition"
        )
        
        show_images = st.checkbox("Show Images", value=True)
        
        st.divider()
        
        st.metric("Total Artworks", len(df))
        st.metric("Departments", df['department'].nunique())
    
    # Main area - Theme input
    st.header("📝 Define Your Exhibitions")
    
    themes = []
    cols = st.columns(min(n_exhibitions, 3))
    
    for i in range(n_exhibitions):
        col_idx = i % 3
        with cols[col_idx]:
            theme = st.text_input(
                f"Exhibition {i+1} Theme",
                placeholder="e.g., Ancient Egypt, Portraits, Religious Art",
                key=f"theme_{i}"
            )
            themes.append(theme)
    
    st.divider()
    
    # Generate button
    if st.button("🎯 Generate Exhibition Recommendations", type="primary"):
        # Filter empty themes
        valid_themes = [t for t in themes if t.strip()]
        
        if not valid_themes:
            st.warning("Please enter at least one exhibition theme.")
            return
        
        # Load recommender
        with st.spinner("Initializing recommendation engine..."):
            recommender = load_recommender(df)
        
        # Generate recommendations
        with st.spinner("Curating exhibitions..."):
            exhibitions = recommender.recommend_exhibitions(
                themes=valid_themes,
                max_pieces_per_exhibition=max_pieces,
                min_pieces_per_exhibition=15
            )
        
        # Display results
        st.success(f"✅ Generated {len(exhibitions)} themed exhibitions!")
        
        for theme, artworks in exhibitions.items():
            st.header(f"🎨 {theme.title()}")
            
            # Exhibition stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Pieces", len(artworks))
            with col2:
                coherence = recommender.evaluate_coherence(artworks['objectID'].tolist())
                st.metric("Coherence Score", f"{coherence:.2f}")
            with col3:
                unique_depts = artworks['department'].nunique()
                st.metric("Departments", unique_depts)
            
            st.divider()
            
            # Display artworks
            for idx, row in artworks.head(10).iterrows():  # Show top 10
                display_artwork(row, show_images)
                st.divider()
            
            if len(artworks) > 10:
                with st.expander(f"View all {len(artworks)} artworks"):
                    st.dataframe(
                        artworks[['objectID', 'title', 'artist', 'department', 'similarity_score']],
                        use_container_width=True
                    )
            
            # Download option
            csv = artworks.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {theme} Exhibition",
                data=csv,
                file_name=f"{theme.replace(' ', '_')}_exhibition.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
    
    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>MET Exhibition AI Curator | INSY 674 Winter 2026</p>
            <p>Data source: Metropolitan Museum of Art Collection API</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
