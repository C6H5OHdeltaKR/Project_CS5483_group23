import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)


def extract_and_print_odds_ratios(best_model, X_columns, ordinal_features, onehot_features, numeric_features):
    """
    Extract logistic regression coefficients, compute Odds Ratios,
    and save a forest plot visualization.
    """
    print("\n[4/5] Extracting feature weights and Odds Ratios from Pipeline...")

    preprocessor = best_model.named_steps['preprocessor']
    ohe_feature_names = preprocessor.named_transformers_['ohe'].get_feature_names_out(onehot_features)
    all_feature_names = ordinal_features + list(ohe_feature_names) + numeric_features

    classifier = best_model.named_steps['classifier']
    coefficients = classifier.coef_[0]
    odds_ratios = np.exp(coefficients)

    or_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient (Log-Odds)': coefficients,
        'Odds Ratio (OR)': odds_ratios
    })

    or_df_sorted = or_df.sort_values(by='Odds Ratio (OR)', ascending=False)

    print("\n" + "=" * 60)
    print(" Top 10 High-Risk Clinical Features")
    print("=" * 60)
    print(or_df_sorted.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print(" Top 5 Protective Clinical Features")
    print("=" * 60)
    print(or_df_sorted.tail(5).to_string(index=False))

    # --- Plot and save Odds Ratio forest plot ---
    print("\n[5/5] Generating and saving Odds Ratio forest plot...")

    top_risk = or_df_sorted.head(10)
    top_protect = or_df_sorted.tail(5)
    plot_df = pd.concat([top_risk, top_protect]).sort_values(by='Odds Ratio (OR)', ascending=True)

    plt.figure(figsize=(10, 8))

    plt.axvline(x=1.0, color='crimson', linestyle='--', lw=2, label='Baseline Risk (OR=1)')
    plt.plot(plot_df['Odds Ratio (OR)'], plot_df['Feature'], 'o', color='teal', markersize=10)

    for index, row in plot_df.iterrows():
        plt.hlines(y=row['Feature'], xmin=1.0, xmax=row['Odds Ratio (OR)'], color='teal', alpha=0.4, lw=2)

    plt.title('Clinical Features Odds Ratios (Top 10 Risk & Top 5 Protective)', fontsize=14)
    plt.xlabel('Odds Ratio (OR) - >1 is Risk, <1 is Protective', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()

    plt.savefig('odds_ratio_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Odds Ratio forest plot saved as 'odds_ratio_forest_plot.png'")

    return or_df_sorted


def train_evaluate_visualize_lr(data_path):
    print(f"Loading preprocessed data: {data_path} ...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: file '{data_path}' not found. Please ensure the preprocessing script has run.")
        return

    target_col = 'diagnosed_diabetes'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    ordinal_features = ['education_level', 'income_level', 'smoking_status']
    education_cats = ['No formal', 'Highschool', 'Graduate', 'Postgraduate']
    income_cats = ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']
    smoking_cats = ['Never', 'Former', 'Current']

    onehot_features = ['gender', 'ethnicity', 'employment_status']
    numeric_features = [col for col in X.columns if col not in ordinal_features + onehot_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', OrdinalEncoder(categories=[education_cats, income_cats, smoking_cats]), ordinal_features),
            ('ohe', OneHotEncoder(drop='first', sparse_output=False), onehot_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='saga', class_weight='balanced', max_iter=2000, random_state=42))
    ])

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'classifier__penalty': ['l1', 'l2'],
        'classifier__C': [0.01, 0.1, 1.0, 10.0]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=kf,
        n_jobs=-1,
        verbose=1
    )

    print("\n[1/5] Running grid search to optimize logistic regression parameters...")
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    best_params = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
    print("Best parameters found:", best_params)

    print("\n[2/5] Running 5-fold cross-validation for objective evaluation metrics...")
    oof_preds = cross_val_predict(best_model, X, y, cv=kf, method='predict', n_jobs=-1)
    oof_probas = cross_val_predict(best_model, X, y, cv=kf, method='predict_proba', n_jobs=-1)[:, 1]

    accuracy = accuracy_score(y, oof_preds)
    precision = precision_score(y, oof_preds)
    sensitivity = recall_score(y, oof_preds)
    f1 = f1_score(y, oof_preds)
    auc_score = roc_auc_score(y, oof_probas)

    tn, fp, fn, tp = confusion_matrix(y, oof_preds).ravel()
    specificity = tn / (tn + fp)

    print("\n=== Logistic Regression Evaluation Results (5-Fold CV) ===")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"AUC:         {auc_score:.4f}")

    print("\n[3/5] Generating and saving ROC curve...")
    plt.figure(figsize=(8, 6))
    fpr, tpr, thresholds = roc_curve(y, oof_probas)
    plt.plot(fpr, tpr, color='crimson', lw=2, label=f'Logistic Regression ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve - Logistic Regression (Early Risk Prediction)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('lr_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> ROC curve saved as 'lr_roc_curve.png'")

    best_model.fit(X, y)

    extract_and_print_odds_ratios(best_model, X.columns, ordinal_features, onehot_features, numeric_features)

    print("\nPipeline execution complete.")
    return best_model


if __name__ == "__main__":
    train_evaluate_visualize_lr('processed_features_for_lr.csv')
