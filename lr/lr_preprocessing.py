import pandas as pd
import numpy as np


def custom_feature_engineering(df):
    """
    Perform early-risk feature extraction and Winsorization on outliers.
    """
    df_engineered = df.copy()

    # 1. Drop leakage columns (avoid target leakage)
    leakage_cols = [
        'diabetes_stage', 'diabetes_risk_score',
        'glucose_fasting', 'glucose_postprandial',
        'insulin_level', 'hba1c'
    ]
    cols_to_drop = [col for col in leakage_cols if col in df_engineered.columns]
    df_engineered = df_engineered.drop(columns=cols_to_drop)

    # 2. Construct advanced clinical derived features
    # Lipid ratios
    df_engineered['tc_hdl_ratio'] = df_engineered['cholesterol_total'] / df_engineered['hdl_cholesterol']
    df_engineered['tg_hdl_ratio'] = df_engineered['triglycerides'] / df_engineered['hdl_cholesterol']

    # Blood pressure derived features
    df_engineered['pulse_pressure'] = df_engineered['systolic_bp'] - df_engineered['diastolic_bp']
    df_engineered['map'] = df_engineered['diastolic_bp'] + df_engineered['pulse_pressure'] / 3

    # Composite lifestyle risk score (0-4)
    df_engineered['lifestyle_risk_score'] = (
            (df_engineered['smoking_status'] == 'Current').astype(int) +
            (df_engineered['physical_activity_minutes_per_week'] < 150).astype(int) +
            (df_engineered['diet_score'] < 5.0).astype(int) +
            (df_engineered['sleep_hours_per_day'] < 6.0).astype(int)
    )

    # 3. Winsorization on numeric features (1%-99% percentile clipping)
    # Logistic regression is sensitive to extreme outliers
    numeric_cols = df_engineered.select_dtypes(include=['float64', 'int64', 'int32']).columns
    target_col = 'diagnosed_diabetes'
    numeric_cols = [col for col in numeric_cols if col != target_col]

    print("Applying 1%-99% Winsorization to numeric columns...")
    for col in numeric_cols:
        lower_bound = df_engineered[col].quantile(0.01)
        upper_bound = df_engineered[col].quantile(0.99)
        df_engineered[col] = df_engineered[col].clip(lower=lower_bound, upper=upper_bound)

    return df_engineered


def main():
    input_file = 'diabetes_dataset.csv'
    output_file = 'processed_features_for_lr.csv'

    print(f"Loading raw data: {input_file} ...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: file '{input_file}' not found. Please ensure it is in the current directory.")
        return

    print("Running feature engineering...")
    df_processed = custom_feature_engineering(df)

    if 'diagnosed_diabetes' not in df_processed.columns:
        print("Warning: target column 'diagnosed_diabetes' not found.")

    print(f"\nSaving processed data to: {output_file} ...")
    df_processed.to_csv(output_file, index=False)
    print("Preprocessing complete. Data is ready for the Logistic Regression pipeline.")


if __name__ == "__main__":
    main()
