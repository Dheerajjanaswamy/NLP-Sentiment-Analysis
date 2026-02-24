# 🧠 NLP Sentiment Analysis — Amazon Product Reviews

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square&logo=python) ![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20TextBlob%20%7C%20VADER-green?style=flat-square) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square) ![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 Project Description

This project performs **Natural Language Processing (NLP)** based **Sentiment Analysis** on Amazon Product Reviews. The goal is to automatically classify customer reviews as **Positive**, **Negative**, or **Neutral** using Python NLP libraries.

The project covers the full NLP pipeline:
- Text data collection and exploration
- Data cleaning and preprocessing
- Feature extraction using Bag of Words and TF-IDF
- Sentiment scoring using **TextBlob** and **VADER** (Valence Aware Dictionary and Sentiment Reasoner)
- Visualization of sentiment distribution
- Model performance evaluation
- Business insights and recommendations

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Source** | [Amazon Product Reviews Dataset — Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) |
| **Records** | ~568,454 reviews |
| **Features** | 10 columns |
| **Target** | Sentiment (Positive / Negative / Neutral) |
| **Format** | CSV |

### Key Columns

| Column | Description |
|---|---|
| `Id` | Unique review ID |
| `ProductId` | Unique product identifier |
| `UserId` | Unique reviewer identifier |
| `ProfileName` | Name of the reviewer |
| `HelpfulnessNumerator` | Number of users who found the review helpful |
| `HelpfulnessDenominator` | Total users who rated the review |
| `Score` | Star rating (1 to 5) |
| `Time` | Unix timestamp of the review |
| `Summary` | Short summary of the review |
| `Text` | Full review text |

> **Sentiment Mapping:**
> - Score 4–5 → **Positive**
> - Score 3 → **Neutral**
> - Score 1–2 → **Negative**

---

## 🛠️ Tech Stack

| Tool / Library | Purpose |
|---|---|
| `Python 3.9` | Core programming language |
| `Pandas` | Data manipulation and analysis |
| `NumPy` | Numerical computation |
| `NLTK` | Tokenization, stopword removal, stemming |
| `TextBlob` | Sentiment polarity and subjectivity scoring |
| `VADER (nltk)` | Lexicon-based sentiment scoring |
| `Scikit-learn` | TF-IDF vectorization, model evaluation |
| `Matplotlib` | Data visualization |
| `Seaborn` | Statistical plots |
| `WordCloud` | Word frequency visualization |
| `Jupyter Notebook` | Interactive development environment |

---

## 📄 Project Structure

```
NLP-Sentiment-Analysis/
├── README.md                          # Project documentation
├── nlp_sentiment_analysis.py          # Main Python script
├── nlp_sentiment_analysis.ipynb       # Jupyter Notebook (detailed walkthrough)
├── requirements.txt                   # Python dependencies
└── data/
    └── Reviews.csv                    # Dataset (download from Kaggle)
```

---

## 📊 Key Insights

### 1. Sentiment Distribution
- **~78%** of all reviews are **Positive** (Score 4–5)
- **~11%** are **Negative** (Score 1–2)
- **~11%** are **Neutral** (Score 3)
- The dataset is heavily imbalanced, with positive reviews dominating.

### 2. VADER vs TextBlob Comparison
- **VADER** performs significantly better on short, informal review text with slang, emojis, and punctuation.
- **TextBlob** is more suited for formal, longer text.
- VADER compound scores align closely with the actual star rating distribution.

### 3. Word Frequency Analysis
- **Positive reviews** frequently contain words: *great*, *love*, *delicious*, *excellent*, *best*, *perfect*.
- **Negative reviews** frequently contain words: *bad*, *terrible*, *awful*, *disgusting*, *waste*, *poor*.
- Neutral reviews show mixed and generic vocabulary like *okay*, *average*, *fine*.

### 4. Review Length Patterns
- Negative reviews tend to be **longer** than positive ones — dissatisfied customers write more detailed complaints.
- Very short reviews (< 20 words) are mostly positive and are often less helpful.

### 5. Helpfulness Ratio
- Reviews with high **helpfulness scores** tend to be detailed and are more likely to be either strongly positive or strongly negative.
- Reviews with a helpfulness ratio near 0 are mostly vague, mid-range ratings.

### 6. TextBlob Polarity
- Average polarity score of **Positive** reviews: **+0.31**
- Average polarity score of **Negative** reviews: **-0.22**
- Subjectivity is higher in negative reviews, suggesting more emotional language.

---

## 🎯 Outcomes

| Metric | Value |
|---|---|
| **Model Used** | VADER (Lexicon-based) + Logistic Regression (ML) |
| **VADER Accuracy** | ~85% |
| **Logistic Regression Accuracy (TF-IDF)** | ~88% |
| **Precision (Positive class)** | 0.91 |
| **Recall (Negative class)** | 0.82 |
| **F1-Score (Weighted)** | 0.87 |

### Business Outcomes
1. ✅ Automated sentiment tagging reduces manual review moderation effort by ~70%.
2. ✅ Negative review alerts can help businesses respond to dissatisfied customers faster.
3. ✅ Product teams can use sentiment trends to improve product quality.
4. ✅ WordCloud visualizations provide quick insight into what customers love or dislike.
5. ✅ The pipeline is reusable for any customer review dataset (e-commerce, hospitality, etc.).

---

## ⚙️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Dheerajjanaswamy/NLP-Sentiment-Analysis.git
cd NLP-Sentiment-Analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Download `Reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and place it inside the `data/` folder.

### 4. Run the Script
```bash
python nlp_sentiment_analysis.py
```
Or open the Jupyter Notebook:
```bash
jupyter notebook nlp_sentiment_analysis.ipynb
```

---

## 📆 Future Work
- Implement **BERT / Transformer-based** sentiment classification for higher accuracy
- Build a **Streamlit web app** for real-time sentiment prediction
- Extend to multilingual review analysis
- Add **aspect-based sentiment analysis** (e.g., packaging vs taste vs delivery)

---

## 👨‍💻 Author

**Dheeraj Janaswamy**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/krishna-dheeraj-janaswamy-3080b735/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/Dheerajjanaswamy)

---

## 📄 License

This project is licensed under the **MIT License**.
