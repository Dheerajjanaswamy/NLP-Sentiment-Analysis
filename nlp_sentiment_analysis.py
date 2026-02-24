
#!/usr/bin/env python3
"""
NLP Sentiment Analysis on Amazon Product Reviews
Author: Dheeraj Janaswamy
Date: February 2026

This script performs sentiment analysis on Amazon product reviews using:
- NLTK for text preprocessing
- TextBlob for sentiment polarity
- VADER for compound sentiment scoring
- Scikit-learn for ML classification
"""

# =============================
# 1. IMPORT LIBRARIES
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# Machine Learning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

print("\u2705 All libraries imported successfully!\n")

# =============================
# 2. LOAD DATASET
# =============================

def load_data(filepath='data/Reviews.csv', sample_size=50000):
    """
    Load Amazon reviews dataset and sample for performance.
    """
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Sample data for faster processing
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}\n")
    return df

# =============================
# 3. EXPLORATORY DATA ANALYSIS
# =============================

def exploratory_analysis(df):
    """
    Perform EDA on the dataset.
    """
    print("="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print("\n1. Dataset Info:")
    print(df.info())
    
    print("\n2. Missing Values:")
    print(df.isnull().sum())
    
    print("\n3. Score Distribution:")
    print(df['Score'].value_counts().sort_index())
    
    # Visualize score distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Score', palette='viridis')
    plt.title('Distribution of Review Scores', fontsize=14, fontweight='bold')
    plt.xlabel('Star Rating')
    plt.ylabel('Count')
    plt.savefig('score_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n4. Sample Reviews:")
    print(df[['Score', 'Summary', 'Text']].head(3))
    print("\n")

# =============================
# 4. TEXT PREPROCESSING
# =============================

def preprocess_text(text):
    """
    Clean and preprocess review text.
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def add_sentiment_labels(df):
    """
    Map star ratings to sentiment categories.
    """
    def map_sentiment(score):
        if score >= 4:
            return 'Positive'
        elif score == 3:
            return 'Neutral'
        else:
            return 'Negative'
    
    df['Sentiment'] = df['Score'].apply(map_sentiment)
    return df

# =============================
# 5. SENTIMENT ANALYSIS
# =============================

def analyze_with_textblob(text):
    """
    Get sentiment polarity using TextBlob.
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity

def analyze_with_vader(text):
    """
    Get sentiment compound score using VADER.
    """
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    return scores['compound']

def apply_sentiment_analysis(df):
    """
    Apply TextBlob and VADER sentiment scoring.
    """
    print("Applying sentiment analysis...")
    
    df['TextBlob_Polarity'] = df['Text'].apply(analyze_with_textblob)
    df['VADER_Compound'] = df['Text'].apply(analyze_with_vader)
    
    print("\u2705 Sentiment scores calculated!\n")
    return df

# =============================
# 6. VISUALIZATION
# =============================

def visualize_sentiments(df):
    """
    Create visualizations for sentiment analysis.
    """
    print("Creating visualizations...")
    
    # Sentiment Distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Sentiment', palette='Set2', order=['Positive', 'Neutral', 'Negative'])
    plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # VADER Compound Score by Sentiment
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x='Sentiment', y='VADER_Compound', palette='coolwarm', order=['Positive', 'Neutral', 'Negative'])
    plt.title('VADER Compound Scores by Sentiment', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment')
    plt.ylabel('VADER Compound Score')
    plt.savefig('vader_scores.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # WordCloud for Positive Reviews
    positive_text = ' '.join(df[df['Sentiment'] == 'Positive']['Processed_Text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(positive_text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - Positive Reviews', fontsize=16, fontweight='bold')
    plt.savefig('wordcloud_positive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\u2705 Visualizations created!\n")

# =============================
# 7. MACHINE LEARNING MODEL
# =============================

def train_model(df):
    """
    Train Logistic Regression model using TF-IDF.
    """
    print("Training machine learning model...")
    
    # Prepare features and target
    X = df['Processed_Text']
    y = df['Sentiment']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Train Logistic Regression
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n\u2705 Model Accuracy: {accuracy*100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['Positive', 'Neutral', 'Negative'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Neutral', 'Negative'], yticklabels=['Positive', 'Neutral', 'Negative'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, tfidf

# =============================
# 8. MAIN EXECUTION
# =============================

def main():
    """
    Main function to run the complete NLP pipeline.
    """
    print("\n" + "="*60)
    print("   NLP SENTIMENT ANALYSIS - AMAZON PRODUCT REVIEWS")
    print("="*60 + "\n")
    
    # Load data
    df = load_data()
    
    # EDA
    exploratory_analysis(df)
    
    # Add sentiment labels
    df = add_sentiment_labels(df)
    
    # Preprocess text
    print("Preprocessing text...")
    df['Processed_Text'] = df['Text'].apply(preprocess_text)
    print("\u2705 Text preprocessing completed!\n")
    
    # Apply sentiment analysis
    df = apply_sentiment_analysis(df)
    
    # Visualizations
    visualize_sentiments(df)
    
    # Train ML model
    model, tfidf = train_model(df)
    
    # Display insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print(f"\n1. Total Reviews Analyzed: {len(df)}")
    print(f"2. Sentiment Distribution:")
    print(df['Sentiment'].value_counts())
    print(f"\n3. Average VADER Scores by Sentiment:")
    print(df.groupby('Sentiment')['VADER_Compound'].mean())
    print(f"\n4. Average TextBlob Polarity by Sentiment:")
    print(df.groupby('Sentiment')['TextBlob_Polarity'].mean())
    print("\n" + "="*60)
    print("\u2705 Analysis Complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
