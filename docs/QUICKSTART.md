# Quick Start Guide

## Initial Setup

### Week 1: Data & Feature Extraction

**Environment Setup:**
```bash
cd "c:\Users\avreu\OneDrive - McGill University\MMA\S3\Winter 1\INSY 674\Final Project\Met_ML_Exhibitions"

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Configure API Keys:**
```bash
# Copy template
copy config\api_keys_template.env config\.env

# Edit config\.env and add Google Vision API key
```

**Feature Extraction:**

Create Jupyter notebooks in `notebooks/`:
- `01_data_exploration.ipynb` - Explore the dataset
- `02_image_analysis.ipynb` - Extract image features with Vision API
- `03_text_analytics.ipynb` - Extract text features

### Week 2: Models

- `04_feature_engineering.ipynb` - Combine features
- `05_clustering_analysis.ipynb` - Test clustering
- `06_recommendation_system.ipynb` - Build recommender

### Week 3: Application

- Refine Streamlit app
- Create evaluation metrics
- Prepare final presentation

---

## Project Management

### GitHub Best Practices

1. **Never commit:**
   - API keys (use .env)
   - Large data files (images, CSVs)
   - Model files (.pkl files)

2. **Al Workflow

**Files to Exclude from Commitsipynb)
   - Documentation (.md files)
   - Configuration templates

**Branch Workflow:**
**Files to Include in Commits
git checkout -b feature/image-features
# Make changes
git add .
git commit -m "Add image feature extraction"
git push origin feature/image-features
# Create Pull Request for team review
```

### Task Management

GitHub Projects can be used to track:
- [ ] Feature extraction complete
- [ ] Models trained
- [ ] App deployed
- [ ] Presentation ready

---

## Common Commands

### Run Streamlit App
```bash
streamlit run src/app/streamlit_app.py
```

### Run Tests
```bash
pytest tests/ -v
```

### Download More Data
```bash
python scripts/download_data.py
```

### Format Code
```bash
black src/ tests/
```

---

## Troubleshooting

**Import errors:**  
Ensure the project root is the working directory and virtual environment is activated.

**API rate limits:**  
Batch processing with delays is implemented in the feature extraction code.

**Memory issues:**  
Process data in chunks and use sampling during development.

---

## Resources

- [MET API Documentation](https://metmuseum.github.io/)
- [Google Vision API Docs](https://cloud.google.com/vision/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
