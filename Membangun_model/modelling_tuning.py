import mlflow
import mlflow.sklearn
import dagshub
import scipy.sparse as sp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Setup DagsHub
dagshub.init(
    repo_owner='ratihayudianurmala',
    repo_name='Eksperimen_SML_Ran',
    mlflow=True
)

# Load data
X_train = sp.load_npz('../preprocessing/olist_preprocessing/X_train.npz')
X_test = sp.load_npz('../preprocessing/olist_preprocessing/X_test.npz')
y_train = pd.read_csv('../preprocessing/olist_preprocessing/y_train.csv').squeeze()
y_test = pd.read_csv('../preprocessing/olist_preprocessing/y_test.csv').squeeze()

# Hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'saga'],
    'class_weight': ['balanced']
}

mlflow.set_experiment("sentiment-analysis-olist-tuning")

with mlflow.start_run(run_name="logistic-regression-tuning"):

    # Grid Search
    base_model = LogisticRegression(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Log params
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("best_C", best_params['C'])
    mlflow.log_param("best_solver", best_params['solver'])
    mlflow.log_param("class_weight", best_params['class_weight'])
    mlflow.log_param("cv_folds", 3)
    mlflow.log_param("scoring", "f1")

    # Log metrics
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
    plt.title('Confusion Matrix - Tuning')
    plt.tight_layout()
    plt.savefig('confusion_matrix_tuning.png')
    mlflow.log_artifact('confusion_matrix_tuning.png')

    # Artefak 2 - Classification Report
    report = classification_report(y_test, y_pred)
    with open('classification_report_tuning.txt', 'w') as f:
        f.write(report)
    mlflow.log_artifact('classification_report_tuning.txt')

    # Artefak 3 - Best Params
    with open('best_params.txt', 'w') as f:
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
    mlflow.log_artifact('best_params.txt')

    # Log model
    mlflow.sklearn.log_model(best_model, "model")

    print(f"Best Params: {best_params}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"AUC-ROC   : {auc:.4f}")
    print("Tuning selesai dan berhasil di-log ke DagsHub!")