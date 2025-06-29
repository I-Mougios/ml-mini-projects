# ML Mini Projects

This repository hosts a collection of self-contained machine learning mini-projects. Each project is organized in its own directory with:

- A dedicated `README.md` describing the problem and methodology
- A Jupyter notebook and/or Python script implementing the solution
- A Conda environment file (`environment.yml`) to ensure reproducibility
- An optional PDF report summarizing findings and results

Each project is designed to be **independent**, meaning you can set up and run each one in isolation using its own virtual environment.

---

## 📁 Projects

### 1. **Network Intrusion Detection**

> Classifies network traffic using the CICIDS2017 dataset into benign and various cyberattack categories such as DoS, DDoS, and brute force.

- **Folder:** `network_intrusion_detection/`
- **Notebook:** `main.ipynb`
- **Environment:** `environment.yml`
- **Details:** Binary and multiclass classification pipeline with feature engineering and model evaluation.

---

### 2. **CLIP Embedding Image Classifier**

> Builds an image classification pipeline using CLIP (Contrastive Language–Image Pre-training) embeddings, combining text and image modalities.

- **Folder:** `clip_embedding_classifier/`
- **Notebook:** `image_cliassification.ipynb`
- **Script:** `main.py` (non-interactive version)
- **Environment:** `environment.yaml`
- **Report:** `project_report.pdf`

---

### 3. **Movie Recommender System**

> Develops a recommender system based on data both from csv files and scraped from IMDb website, exploring both content-based and collaborative filtering techniques.

- **Folder:** `movie_recommender/`
- **Notebook:** `main.ipynb`
- **Data Scraper:** `scrape_imdb_data.ipynb`
- **Environment:** `environment.yml`
- **Report:** `project_report.pdf`
- **CSV files: `https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset`

---

## 🛠 Setup Instructions

Each project includes a `environment.yml` file. You can create the environment with:

```bash
conda env create -f path/to/envi