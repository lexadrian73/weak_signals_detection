# Weak Signals Detection in Strategic Monitoring

This project focuses on detecting **weak signals** using Natural Language Processing, topic modeling, time series analysis, and change point detection. It is designed for strategic foresight and early warning in dynamic environments such as media monitoring and policy analysis.

## 📁 Project Structure

| Notebook                        | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `1_Data Preprocessing.ipynb`     | Loads and preprocesses raw news data, performs tokenization and cleaning.  |
| `2_Exploratory_Data_Analysis.ipynb` | Explores the frequency, volatility, and behavior of keyword emergence.    |
| `3_KEM_KIM.ipynb`                | Builds Keyword Emergence Map (KEM) and Keyword Issue Map (KIM).            |
| `4_bert_topics.ipynb`            | Trains BERTopic models and detects semantically weak topics.               |
| `5_time_series_dm.ipynb`         | Builds keyword frequency time series for each topic.                       |
| `6_time_series_breakpoints.ipynb`| Detects change points in time series using Prophet and Ruptures.           |
| `lda_test_model.ipynb`           | Trains and evaluates traditional LDA models for topic comparison.          |

## 🔍 Objectives

- Identify emerging or weak signals from large corpora of news articles.
- Model keyword evolution using topic modeling (BERTopic and LDA).
- Detect ruptures and temporal shifts in topic relevance using Prophet and Ruptures.
- Provide interpretable visualizations for strategic monitoring.

## 🧰 Tools & Libraries

- Python 3.10+
- `BERTopic`, `transformers`, `gensim`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`
- `Prophet` for trend and changepoint detection
- `ruptures` for time series segmentation
- `pandas`, `numpy`, `statsmodels`, `spacy` (for text processing)

## ⚙️ Setup

```bash
git clone https://github.com/lexadrian73/weak_signals_detection.git
cd weak_signals_detection
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
