# Data Science Best Practices

This document outlines the best practices followed in this project, aligned with enterprise data science standards.

---

## 1. Project Organization

### Directory Structure
Following the industry-standard structure for data science projects:

```
project/
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned, transformed data
│   └── results/      # Model outputs, predictions
├── src/              # Source code
│   ├── data/         # Data loading and validation
│   ├── features/     # Feature engineering
│   ├── models/       # ML models
│   └── app/          # Application code
├── notebooks/        # Jupyter notebooks for exploration
├── tests/            # Unit and integration tests
├── config/           # Configuration files
├── docs/             # Documentation
└── scripts/          # Standalone scripts
```

**Benefits:**
- Clear separation of concerns
- Easy navigation for team members
- Scalable as project grows
- Standard structure recognized across industry

---

## 2. Version Control

### Git Best Practices

**Always Use .gitignore:**
- Exclude large data files (images, CSVs)
- Exclude credentials and API keys
- Exclude virtual environments
- Exclude temporary files and caches

**Commit Messages:**
- Use descriptive, actionable commit messages
- Format: `verb + what changed + why (if not obvious)`
- Example: `Add image feature extraction using Google Vision API`

**Branching Strategy:**
- `main` - production-ready code
- `develop` - integration branch
- `feature/*` - new features
- `bugfix/*` - bug fixes

**Example Workflow:**
```bash
git checkout -b feature/vision-api
# Make changes
git add .
git commit -m "Implement Google Vision API integration for image analysis"
git push origin feature/vision-api
# Create Pull Request
```

---

## 3. Code Quality

### Modular Code Design

**Principles:**
- Single Responsibility: Each function does one thing well
- DRY (Don't Repeat Yourself): Reuse code through functions/classes
- Clear naming: Functions and variables have descriptive names

**Example Structure:**
```python
# Good: Modular, reusable
def load_data(filepath):
    """Load and validate data."""
    pass

def preprocess_text(text):
    """Clean and normalize text."""
    pass

def extract_features(data):
    """Extract ML features."""
    pass
```

### Documentation

**Docstrings for All Functions:**
```python
def recommend_exhibitions(themes, max_pieces=30):
    """
    Generate exhibition recommendations based on themes.
    
    Parameters
    ----------
    themes : List[str]
        List of exhibition themes
    max_pieces : int, optional
        Maximum artworks per exhibition (default: 30)
        
    Returns
    -------
    dict
        Dictionary mapping theme to recommended artworks
    """
    pass
```

**Type Hints:**
```python
from typing import List, Dict
import pandas as pd

def process_images(image_paths: List[str]) -> pd.DataFrame:
    """Process images and extract features."""
    pass
```

### Code Style

**Use Linters:**
- `black` - Code formatter
- `flake8` - Style guide enforcement  
- `pylint` - Code analysis

**Configuration:**
```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Run linter
pylint src/
```

---

## 4. Dependency Management

### Requirements Files

**requirements.txt:**
- Pin exact versions for reproducibility
- Include all direct dependencies
- Keep up to date

```txt
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
```

**Why Pin Versions:**
- Ensures consistency across environments
- Prevents breaking changes from updates
- Makes debugging easier

### Virtual Environments

**Always Use Virtual Environments:**
```bash
# Create environment
python -m venv venv

# Activate
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

**Benefits:**
- Isolation from other projects
- No version conflicts
- Easy to recreate
- Team consistency

---

## 5. Configuration Management

### Separate Configuration from Code

**Use config.yaml:**
```yaml
model:
  n_clusters: 15
  random_state: 42

features:
  vision_weight: 0.5
  text_weight: 0.5
```

**Load Configuration:**
```python
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

n_clusters = config['model']['n_clusters']
```

**Benefits:**
- Change parameters without code changes
- Easy experimentation
- Clear documentation of settings

### Secrets Management

**Never commit API keys:**
```bash
# .env file (in .gitignore)
GOOGLE_VISION_API_KEY=abc123xyz

# Use python-dotenv
from dotenv import load_dotenv
import os

load_dotenv('config/.env')
api_key = os.getenv('GOOGLE_VISION_API_KEY')
```

---

## 6. Testing

### Unit Tests

**Test Critical Functions:**
```python
import pytest

def test_data_loader():
    """Test that data loads correctly."""
    df = load_met_data()
    assert len(df) > 0
    assert 'objectID' in df.columns

def test_recommender():
    """Test recommendation engine."""
    recommender = ExhibitionRecommender(features, metadata)
    results = recommender.recommend_for_theme('egyptian')
    assert len(results) > 0
```

**Run Tests:**
```bash
pytest tests/ -v
pytest tests/ --cov=src  # With coverage
```

**Benefits:**
- Catch bugs early
- Prevent regressions
- Enable refactoring confidence
- Document expected behavior

---

## 7. Data Management

### Data Versioning

**Track Data Changes:**
- Document data sources
- Record collection dates
- Note preprocessing steps
- Version processed datasets

**Data Documentation:**
```markdown
## Dataset: MET Artworks
- **Source:** Metropolitan Museum of Art API
- **Collection Date:** February 2026
- **Size:** 448 artworks
- **Format:** CSV + JPEG images
- **Preprocessing:** Downloaded via API, filtered for images
```

### Data Privacy

**Protected Information:**
- API keys and credentials
- Large datasets (use data storage, not Git)
- Personal information (if applicable)
- Proprietary data

---

## 8. Reproducibility

### Random Seeds

**Set Seeds for Reproducibility:**
```python
import random
import numpy as np

# Set seeds
random.seed(42)
np.random.seed(42)

# In scikit-learn
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
```

### Environment Documentation

**Document Environment:**
- Python version
- Operating system
- Hardware requirements
- External dependencies

**In README:**
```markdown
## Requirements
- Python 3.9+
- 8GB RAM minimum
- Google Cloud Vision API account
```

---

## 9. Experiment Tracking

### Log Experiments

**Track Model Experiments:**
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("n_clusters", 15)
    mlflow.log_param("similarity_metric", "cosine")
    mlflow.log_metric("silhouette_score", score)
    mlflow.log_artifact("model.pkl")
```

**Benefits:**
- Compare different approaches
- Reproduce best results
- Share findings with team
- Document what works/doesn't work

---

## 10. Code Review

### Pull Request Process

**Before Merging:**
1. Code runs without errors
2. Tests pass
3. Documentation updated
4. Code reviewed by teammate
5. No merge conflicts

**Review Checklist:**
- [ ] Code is readable and well-documented
- [ ] No hardcoded values (use config)
- [ ] Tests included for new functionality
- [ ] No API keys or secrets committed
- [ ] Follows project structure

---

## 11. Continuous Integration (Optional)

### Automated Testing

**GitHub Actions Example:**
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/
```

**Benefits:**
- Automatic test execution
- Catch issues before merge
- Maintain code quality

---

## 12. Documentation

### README.md

**Essential Sections:**
- Project overview and goals
- Installation instructions
- Usage examples
- Team members and contact
- License information

### Code Documentation

**Inline Comments:**
- Explain WHY, not WHAT
- Document complex logic
- Add TODO notes for future work

```python
# Good: Explains reasoning
# Use cosine similarity because it's scale-invariant
similarity = cosine_similarity(features)

# Bad: States the obvious
# Calculate similarity
similarity = cosine_similarity(features)
```

---

## 13. Performance Optimization

### Profiling

**Identify Bottlenecks:**
```python
import cProfile

cProfile.run('extract_features(images)')
```

### Caching

**Cache Expensive Operations:**
```python
import functools

@functools.lru_cache(maxsize=128)
def expensive_computation(param):
    # Cached result on repeated calls
    pass
```

---

## 14. Collaboration

### Team Communication

**Tools:**
- GitHub Issues for bugs and features
- Pull Requests for code review
- Project boards for task tracking
- README for quick reference

**Best Practices:**
- Clear, descriptive issue titles
- Reference issues in commits (`fixes #42`)
- Regular sync meetings
- Document decisions

---

## Summary Checklist

**Project Setup:**
- [ ] Proper directory structure
- [ ] .gitignore configured
- [ ] requirements.txt with pinned versions
- [ ] Virtual environment created
- [ ] README.md completed

**Code Quality:**
- [ ] Modular, reusable functions
- [ ] Docstrings on all functions
- [ ] Type hints where applicable
- [ ] No hardcoded values
- [ ] Linter-compliant code

**Data & Config:**
- [ ] Configuration separated from code
- [ ] API keys in .env (gitignored)
- [ ] Data files gitignored
- [ ] Data sources documented

**Testing & CI:**
- [ ] Unit tests for critical functions
- [ ] Tests pass before commits
- [ ] CI/CD configured (optional)

**Documentation:**
- [ ] README with setup instructions
- [ ] Code comments for complex logic
- [ ] API documentation
- [ ] Usage examples

**Version Control:**
- [ ] Meaningful commit messages
- [ ] Feature branches
- [ ] Pull requests for major changes
- [ ] Code reviews completed

---

## References

- [Python Package Structure](https://docs.python-guide.org/writing/structure/)
- [PEP 8 Style Guide](https://pep8.org/)
- [scikit-learn Best Practices](https://scikit-learn.org/stable/developers/contributing.html)
- [The Twelve-Factor App](https://12factor.net/)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
