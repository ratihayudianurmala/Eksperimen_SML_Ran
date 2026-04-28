import mlflow
import mlflow.sklearn
import dagshub
import scipy.sparse as sp
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Setup DagsHub
dagshub.init(
    repo_owner='ratihayudianurmala',
    repo_name='Eksperimen_SML_Ran',
    mlflow=True
)

# Load data preprocessed
X_train = sp.load_npz('../preprocessing/olist_preprocessing/X_train.npz')
X_test = sp.load_npz('../preprocessing/olist_preprocessing/X_test.npz')
y_train = pd.read_csv('../preprocessing/olist_preprocessing/y_train.csv').squeeze()
y_test = pd.read_csv('../preprocessing/olist_preprocessing/y_test.csv').squeeze()

# Set experiment
mlflow.set_experiment("sentiment-analysis-olist")

with mlflow.start_run(run_name="logistic-regression-baseline"):

    # Model
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Manual logging - params
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("random_state", 42)

    # Manual logging - metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", auc)

    # Artefak 1 - Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negatif', 'Positif'],
                yticklabels=['Negatif', 'Positif'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    # Artefak 2 - Classification Report
    report = classification_report(y_test, y_pred)
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    mlflow.log_artifact('classification_report.txt')

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy  : {acc:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"AUC-ROC   : {auc:.4f}")
    print("Model dan artefak berhasil di-log ke DagsHub!")