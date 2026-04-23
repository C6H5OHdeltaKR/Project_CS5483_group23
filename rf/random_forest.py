import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import time


def train_evaluate_visualize(data_path):
    print(f"Loading data: {data_path} ...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: file '{data_path}' not found.")
        return

    target_col = 'diagnosed_diabetes'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 1. Configure Random Forest and cross-validation strategy
    rf_base = RandomForestClassifier(class_weight='balanced_subsample', random_state=42)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. Grid search for optimal parameters
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_leaf': [20, 50]
    }

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        scoring='f1',
        cv=kf,
        n_jobs=-1,
        verbose=1
    )

    print("\n[1/4] Running grid search to optimize model parameters...")
    start_time = time.time()
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    print(f"Grid search complete. Elapsed: {(time.time() - start_time) / 60:.2f} min")
    print("Best parameters:", grid_search.best_params_)

    # 3. Out-of-fold cross-validation predictions
    print("\n[2/4] Running 5-fold cross-validation for evaluation metrics...")
    oof_preds = cross_val_predict(best_model, X, y, cv=kf, method='predict', n_jobs=-1)
    oof_probas = cross_val_predict(best_model, X, y, cv=kf, method='predict_proba', n_jobs=-1)[:, 1]

    # 4. Compute six core evaluation metrics
    print("\n[3/4] Computing model performance metrics...")
    accuracy = accuracy_score(y, oof_preds)
    precision = precision_score(y, oof_preds)
    sensitivity = recall_score(y, oof_preds)
    f1 = f1_score(y, oof_preds)
    auc_score = roc_auc_score(y, oof_probas)

    tn, fp, fn, tp = confusion_matrix(y, oof_preds).ravel()
    specificity = tn / (tn + fp)

    print("\n=== Evaluation Results (5-Fold CV) ===")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"AUC:         {auc_score:.4f}")

    # 5. Visualization
    print("\n[4/4] Generating and saving visualization charts...")

    # --- Chart 1: ROC Curve ---
    plt.figure(figsize=(8, 6))
    fpr, tpr, thresholds = roc_curve(y, oof_probas)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> ROC curve saved as 'roc_curve.png'")

    # --- Chart 2: Feature Importance ---
    best_model.fit(X, y)
    importances = best_model.feature_importances_

    feature_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
    plt.title('Random Forest Feature Importance (Early Risk Prediction)')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.grid(axis='x', alpha=0.3)
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Feature importance chart saved as 'feature_importance.png'")

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    train_evaluate_visualize('rf_processed_diabetes_data.csv')
