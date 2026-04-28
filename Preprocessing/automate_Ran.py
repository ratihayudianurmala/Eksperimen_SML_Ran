import pandas as pd
import numpy as np
import re
import os
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def load_data(raw_path):
    reviews = pd.read_csv(f'{raw_path}/olist_order_reviews_dataset.csv')
    return reviews


def create_labels(df):
    df = df[df['review_score'] != 3].copy()
    df['sentiment'] = df['review_score'].apply(lambda x: 1 if x >= 4 else 0)
    return df


def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-záéíóúàãõâêôç\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess(df):
    df['clean_comment'] = df['review_comment_message'].apply(clean_text)
    return df


def vectorize_and_split(df):
    tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95)
    X = tfidf.fit_transform(df['clean_comment'])
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def save_output(X_train, X_test, y_train, y_test, output_path):
    os.makedirs(output_path, exist_ok=True)
    sp.save_npz(f'{output_path}/X_train.npz', X_train)
    sp.save_npz(f'{output_path}/X_test.npz', X_test)
    y_train.to_csv(f'{output_path}/y_train.csv', index=False)
    y_test.to_csv(f'{output_path}/y_test.csv', index=False)
    print(f"Preprocessing selesai! File disimpan di {output_path}")


def main():
    raw_path = 'data_raw'
    output_path = 'preprocessing/olist_preprocessing'

    print("Loading data...")
    reviews = load_data(raw_path)

    print("Creating labels...")
    df = create_labels(reviews)

    print("Cleaning text...")
    df = preprocess(df)

    print("Vectorizing dan splitting...")
    X_train, X_test, y_train, y_test = vectorize_and_split(df)

    print("Saving output...")
    save_output(X_train, X_test, y_train, y_test, output_path)


if __name__ == '__main__':
    main()