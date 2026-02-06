# MET Exhibition AI Curator 🎨


> An AI-powered recommendation system to help Metropolitan Museum of Art exhibition planners automatically curate thematic exhibitions, reducing manual curation effort.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Solution](#solution)
- [Team](#team)
- [Dataset](#dataset)
- [Technical Approach](#technical-approach)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

This project addresses the growing operational challenges at the Metropolitan Museum of Art, where workers are facing increased workload due to rising museum popularity. We developed an AI-powered chatbot that automatically recommends optimal artwork groupings for themed exhibitions based on computer vision and natural language processing.

**Key Features:**
- 🤖 Interactive Streamlit chatbot interface
- 🖼️ Image analysis using Google Vision API
- 📊 Text analytics on artwork metadata
- 🎯 Content-based recommendation engine
- ✅ Constraint handling (exhibition size, diversity, theme coherence)

---

## 💼 Business Problem

**Challenge:** MET exhibition planners spend countless hours manually curating each exhibition, leading to:
- Worker burnout and unionization efforts
- Slow exhibition development cycles
- Limited exploration of creative curatorial options
- High operational costs

**Our Approach:** Build an automated recommendation system using Google Vision API for image analysis and natural language processing to suggest optimal artwork groupings for themed exhibitions.

---

## 💡 Solution

An intelligent recommendation system where curators input:
- Number of exhibitions needed
- Themes per exhibition (e.g., "ancient Egypt", "religious art", "portraits")
- Maximum pieces per exhibition (20-50)

**Output:** Ranked lists of artworks per exhibition with similarity scores and visual previews.

---

## 👥 Team

| Name | GitHub ID | Role |
|------|-----------|------|
| Sofia Berumen | @sofiaberumenr | NLP & Text Analytics - TF-IDF, LDA, text preprocessing, metadata analysis |
| Zoe Levings | @zoe-levings | App Development & Integration - Streamlit UI, visualization, system integration, deployment |
| Romane Lucas-Girardville | @romane-lg | ML & Recommendation Engine - Feature engineering, similarity scoring, clustering, evaluation |
| Andrea Vreugdenhil | @andreavreug | Vision API & Image Features - Google Cloud setup, API integration, image processing pipeline |

**Repository:** [romane-lg/Met_ML_Exhibitions](https://github.com/romane-lg/Met_ML_Exhibitions)

---

## 📊 Dataset

**Source:** Metropolitan Museum of Art Collection API  
**Size:** 448 artworks with images and metadata  
**Coverage:** Diverse departments (Egyptian Art, European Paintings, Medieval Art, etc.)

**Metadata Fields:**
- `objectID`: Unique identifier
- `title`: Artwork title
- `artist`: Artist name
- `department`: Museum department
- `objectDate`: Creation date/period
- `medium`: Materials and techniques
- `image_path`: Local path to downloaded image

**Data Collection:**
```bash
python scripts/download_data.py
```

---

## 🔬 Technical Approach

### **1. Feature Engineering**

#### **Image Features (Computer Vision)**
- Google Vision API for label detection, object recognition
- Color analysis (dominant colors, palettes)
- Text detection (inscriptions, signatures)
- Web entities for contextual understanding

#### **Text Features (NLP)**
- TF-IDF vectorization on combined text (title + artist + medium)
- Topic modeling (Latent Dirichlet Allocation)
- Named Entity Recognition for dates, locations, artists
- Sentiment analysis on artwork descriptions

### **2. Recommendation Engine**

- **Content-based filtering:** Cosine similarity on combined feature vectors
- **Clustering:** K-means to discover natural thematic groups
- **Query expansion:** Semantic search to match user themes
- **Constraint optimization:** Balance theme coherence with exhibition size limits

### **3. Tech Stack**

| Component | Technology |
|-----------|-----------|
| Computer Vision | Google Vision API |
| NLP | scikit-learn, spaCy, NLTK |
| ML Modeling | scikit-learn, numpy, pandas |
| Frontend | Streamlit |
| Data Viz | matplotlib, seaborn, plotly |
| Version Control | Git, GitHub |
| Experiment Tracking | MLflow (optional) |

---

## 📁 Project Structure

```
Met_ML_Exhibitions/
├── data/
│   ├── raw/                          # Original data (gitignored)
│   │   ├── met_data.csv
│   │   └── images/
│   ├── processed/                     # Processed features
│   │   ├── vision_features.pkl
│   │   ├── text_features.pkl
│   │   └── combined_embeddings.pkl
│   └── results/                       # Model outputs
│       └── exhibition_recommendations.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_image_analysis.ipynb
│   ├── 03_text_analytics.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_clustering_analysis.ipynb
│   └── 06_recommendation_system.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py            # Data loading utilities
│   ├── features/
│   │   ├── __init__.py
│   │   ├── image_features.py         # Google Vision API
│   │   ├── text_features.py          # NLP preprocessing
│   │   └── feature_engineering.py    # Feature combination
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clustering.py             # Clustering algorithms
│   │   ├── recommender.py            # Recommendation engine
│   │   └── utils.py                  # Helper functions
│   └── app/
│       ├── __init__.py
│       └── streamlit_app.py          # Web interface
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   └── test_recommender.py
├── config/
│   ├── config.yaml                   # Configuration parameters
│   └── api_keys_template.env         # API key template
├── docs/
│   ├── presentation.md               # Presentation outline
│   └── project_report.md             # Final report
├── scripts/
│   ├── download_data.py              # MET API data collection
│   ├── extract_features.py           # Feature extraction pipeline
│   └── train_models.py               # Model training script
├── .gitignore
├── README.md
├── requirements.txt
├── environment.yml
└── LICENSE
```

---

## 🚀 Installation

### **Prerequisites**
- Python 3.9 or higher
- Google Cloud account (for Vision API)
- Git

### **Setup**

1. **Clone the repository:**
```bash
git clone https://github.com/romane-lg/Met_ML_Exhibitions.git
cd Met_ML_Exhibitions
```

2. **Create virtual environment:**
```bash
# Using conda
conda env create -f environment.yml
conda activate met-curator

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set up API keys:**
```bash
# Copy the template
cp config/api_keys_template.env config/.env

# Edit config/.env and add your Google Vision API key
# GOOGLE_VISION_API_KEY=your_api_key_here
```

4. **Download data (if not already done):**
```bash
python scripts/download_data.py
```

---

## 💻 Usage

### **1. Run Exploratory Data Analysis**
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### **2. Extract Features**
```bash
# Extract image features with Google Vision API
python scripts/extract_features.py --type image

# Extract text features
python scripts/extract_features.py --type text
```

### **3. Train Models**
```bash
python scripts/train_models.py --config config/config.yaml
```

### **4. Launch Streamlit App**
```bash
streamlit run src/app/streamlit_app.py
```

Open your browser at `http://localhost:8501`

### **5. Run Tests**
```bash
pytest tests/ -v
```

---

## 📈 Results

*(To be updated after model training)*

**Expected Metrics:**
- **Clustering Quality:** Silhouette Score > 0.5
- **Recommendation Relevance:** NDCG@10 > 0.7
- **Time Savings:** 60%+ reduction in manual curation time
- **User Satisfaction:** Survey scores (if applicable)

**Sample Exhibition Output:**
```
Exhibition 1: Ancient Egyptian Artifacts (25 pieces)
- Sarcophagus of Henhenet (Similarity: 0.95)
- Book of the Dead of Imhotep (Similarity: 0.92)
- Statuette of Amenhotep I (Similarity: 0.89)
...
```

---

## 🤝 Contributing

This is an academic project for INSY 674 - Enterprise Data Science at McGill University.

**Development Workflow:**
1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and commit: `git commit -m "Add feature"`
3. Push to branch: `git push origin feature/your-feature`
4. Create Pull Request for team review

**Best Practices:**
- Write docstrings for all functions
- Add unit tests for new features
- Update documentation as needed
- Never commit API keys or large data files

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---


**Last Updated:** February 5, 2026  
**Course:** INSY 674 - Enterprise Data Science, Winter 2026  
**Institution:** McGill University - Desautels Faculty of Management
