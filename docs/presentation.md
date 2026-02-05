# MET Exhibition AI Curator - Presentation Outline

**Team:** [Your Team Name]  
**Members:** [Names & GitHub IDs]  
**Date:** January 30, 2026  
**Duration:** 5 minutes

---

## Slide 1: Title & Team (30 sec)

### MET Exhibition AI Curator 🎨
**Reducing Exhibition Curation Time by 60% with AI**

**Team Members:**
- [Name 1] - [@github-id-1] - Data Scientist / ML Engineer
- [Name 2] - [@github-id-2] - NLP Specialist
- [Name 3] - [@github-id-3] - Computer Vision Engineer

**GitHub Repository:** [romane-lg/Met_ML_Exhibitions](https://github.com/romane-lg/Met_ML_Exhibitions)

---

## Slide 2: Context & Business Problem (1 min)

### The Challenge
- **MET workers are being unionized** due to increasing workload
- Museum popularity has **grown 40% over past 5 years**
- Exhibition curation takes **40+ hours per exhibition**
- Manual process is **costly, slow, and exhausting**

### Current State
- Curators manually review thousands of artworks
- Limited time to explore creative themes
- Risk of missing thematic connections
- High operational costs

### Our Solution
AI-powered recommendation system that suggests optimal artwork groupings based on themes

---

## Slide 3: Hypothesis (30 sec)

### Primary Hypothesis
**AI-driven automation can reduce curation time by 60%+ while maintaining or improving exhibition quality**

### Supporting Hypotheses
1. **Computer Vision** can identify visual patterns humans might miss
2. **NLP** can extract thematic connections from metadata
3. **Combining both** creates better recommendations than either alone
4. **Constraint optimization** ensures practical, balanced exhibitions

---

## Slide 4: Data (1 min)

### Dataset Overview
**Source:** Metropolitan Museum of Art Collection API  
**Size:** 450 artworks (scalable to full collection)

### What We Have:
✅ **Images:** High-quality photos of each artwork  
✅ **Metadata:**
   - Title, Artist, Department
   - Creation date, Medium, Dimensions
   - Historical period, Cultural origin

### Data Diversity:
- **10+ departments:** Egyptian Art, European Paintings, Medieval Art, Asian Art, etc.
- **Time span:** 3000 BCE to 21st century
- **Geographic range:** Global collection
- **Media types:** Paintings, sculptures, textiles, ceramics, etc.

**Perfect for testing thematic recommendations!**

---

## Slide 5: Technical Approach - Week by Week (1.5 min)

### 📅 Week 1: Feature Engineering
**Computer Vision (Google Vision API)**
- Label detection (objects, concepts)
- Color analysis (palettes, dominant colors)
- Text detection (inscriptions)
- Web entities (contextual info)

**Natural Language Processing**
- TF-IDF vectorization on combined text
- Topic modeling (LDA) for latent themes
- Named Entity Recognition (dates, places)
- Metadata encoding (departments, periods)

**Output:** Combined feature vectors for each artwork

---

### 📅 Week 2: ML Models & Recommendation Engine

**Clustering Analysis**
- K-means to discover natural groupings
- Silhouette analysis for validation
- DBSCAN for outlier detection

**Recommendation System**
- Content-based filtering using cosine similarity
- Query expansion (user input → semantic search)
- Constraint handling:
  - Max/min pieces per exhibition
  - Department diversity
  - No artwork duplication across exhibitions

**Output:** Trained recommender model

---

### 📅 Week 3: Application & Evaluation

**Streamlit Web Interface**
- Chat-like input: "I want 5 exhibitions with these themes..."
- Visual display: images + metadata
- Export: CSV/PDF reports

**Evaluation Metrics**
- Coherence scores (within-exhibition similarity)
- Diversity metrics (department distribution)
- User study (if time permits)

**Deliverables:**
- Working demo
- GitHub repository
- Final presentation

---

## Slide 6: Tools & Technologies (30 sec)

### Tech Stack

| Component | Technology |
|-----------|-----------|
| **Computer Vision** | Google Vision API |
| **NLP** | scikit-learn, spaCy, NLTK |
| **ML** | scikit-learn, numpy, pandas |
| **Frontend** | Streamlit |
| **Visualization** | matplotlib, seaborn, plotly |
| **Version Control** | Git, GitHub, GitHub Projects |

### Best Practices
✅ Modular code structure  
✅ Unit tests (pytest)  
✅ Documentation (docstrings, README)  
✅ Environment management (requirements.txt)  
✅ API key security (.env, .gitignore)

---

## Slide 7: Expected Impact & Next Steps (30 sec)

### Expected Outcomes
📊 **60%+ reduction** in curation time  
🎯 **Higher quality** theme matching  
💡 **Discovery** of novel artwork connections  
💰 **Cost savings** for museum operations

### Scalability
- Start with 450 artworks (Proof of Concept)
- Scale to full MET collection (450,000+ objects)
- Adapt to other museums

### Next Steps (Post-Presentation)
1. Complete feature extraction (Week 1)
2. Train and evaluate models (Week 2)
3. Deploy Streamlit app (Week 3)
4. Final presentation & demo

---

## Backup Slides

### Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| Google Vision API costs | Free tier (1000/mo), cache results |
| Small dataset | Focus on POC, discuss scalability |
| Integration complexity | Start simple, iterate |
| Team coordination | GitHub Projects, weekly standups |

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Coherence | >0.6 | Avg pairwise similarity |
| Coverage | >90% | % artworks used |
| Diversity | >5 depts/exhibition | Unique departments |
| Time savings | >60% | Compared to manual |

---

## Q&A Preparation

**Expected Questions:**

1. **"Why only 450 artworks?"**
   - POC for 3-week timeline
   - Demonstrates feasibility before scaling
   - Sufficient diversity for testing

2. **"How do you handle API costs?"**
   - Google Vision free tier: 1000 requests/month
   - Batch processing & caching
   - One-time extraction, reuse features

3. **"What if recommendations aren't good enough?"**
   - Human-in-the-loop: curators refine results
   - A/B testing different algorithms
   - Continuous improvement based on feedback

4. **"How is this different from search?"**
   - Proactive recommendations vs reactive search
   - Considers exhibition-level constraints
   - Balances similarity with diversity

---

## Speaker Notes

### Opening (You)
"Good morning! We're [Team Name], and we're tackling a real problem at the Metropolitan Museum of Art..."

### Transitions
- Slide 2→3: "So our hypothesis is that..."
- Slide 4→5: "Now let's talk about how we'll actually build this..."
- Slide 6→7: "Finally, what impact do we expect?"

### Closing
"We're excited to build this and will keep you updated throughout the project. Happy to answer any questions!"

### Time Management
- Use timer
- Practice 3x to stay under 5 minutes
- Prepare to skip backup slides if time tight

---

## Visual Elements to Include

1. **MET Logo** (if permitted)
2. **Sample artwork images** from dataset
3. **Architecture diagram** (data flow)
4. **Screenshot mockup** of Streamlit app
5. **Team photo** (optional)

---

**Good luck with your presentation! 🚀**
