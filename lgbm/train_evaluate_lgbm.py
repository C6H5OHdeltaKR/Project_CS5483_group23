import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import lightgbm as lgb


def train_evaluate_visualize_lgbm(data_path):
    print(f"Loading preprocessed data: {data_path} ...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: file '{data_path}' not found. Please ensure the preprocessing script has run successfully.")
        return

    # --- 1. Restore nominal categorical features (LightGBM optimization) ---
    print("Casting nominal categorical features to 'category' type...")
    nominal_features = ['gender', 'ethnicity', 'employment_status']
    for col in nominal_features:
        if col in df.columns:
            df[col] = df[col].astype('category')

    target_col = 'diagnosed_diabetes'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --- 2. Configure LightGBM model and cross-validation strategy ---
    lgb_base = lgb.LGBMClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- 3. Build parameter grid ---
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63]
    }

    grid_search = GridSearchCV(
        estimator=lgb_base,
        param_grid=param_grid,
        scoring='f1',
        cv=kf,
        n_jobs=-1,
        verbose=1
    )

    # --- 4. Run grid search ---
    print("\n[1/4] Running grid search to optimize LightGBM parameters...")
    start_time = time.time()

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    print(f"Grid search complete. Elapsed: {(time.time() - start_time) / 60:.2f} min")
    print("Best parameters found:", grid_search.best_params_)

    # --- 5. Out-of-fold cross-validation predictions ---
    print("\n[2/4] Running 5-fold cross-validation for objective evaluation metrics...")
    oof_preds = cross_val_predict(best_model, X, y, cv=kf, method='predict', n_jobs=-1)
    oof_probas = cross_val_predict(best_model, X, y, cv=kf, method='predict_proba', n_jobs=-1)[:, 1]

    # --- 6. Compute six core evaluation metrics ---
    accuracy = accuracy_score(y, oof_preds)
    precision = precision_score(y, oof_preds)
    sensitivity = recall_score(y, oof_preds)
    f1 = f1_score(y, oof_preds)
    auc_score = roc_auc_score(y, oof_probas)

    tn, fp, fn, tp = confusion_matrix(y, oof_preds).ravel()
    specificity = tn / (tn + fp)

    print("\n=== LightGBM Evaluation Results (5-Fold CV) ===")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"AUC:         {auc_score:.4f}")

    # --- 7. Plot and save ROC curve ---
    print("\n[3/4] Generating and saving ROC curve...")
    plt.figure(figsize=(8, 6))
    fpr, tpr, thresholds = roc_curve(y, oof_probas)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'LightGBM ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve - LightGBM (Early Risk Prediction)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('lgbm_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> ROC curve saved as 'lgbm_roc_curve.png'")

    # --- 8. Plot and save feature importance ---
    print("\n[4/4] Generating and saving feature importance chart...")
    best_model.fit(X, y)

    plt.figure(figsize=(10, 8))
    lgb.plot_importance(best_model, max_num_features=15, importance_type='gain',
                        title='LightGBM Feature Importance (Information Gain)',
                        xlabel='Feature Importance (Gain)',
                        ylabel='Features',
                        figsize=(10, 8), color='steelblue')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('lgbm_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Feature importance chart saved as 'lgbm_feature_importance.png'")

    print("\nPipeline execution complete.")
    return best_model


if __name__ == "__main__":
    train_evaluate_visualize_lgbm('processed_features_for_lgbm.csv')
