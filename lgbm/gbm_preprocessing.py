import pandas as pd
import numpy as np


def lgbm_feature_engineering(df):
    df_engineered = df.copy()

    # 1. Drop leakage columns
    leakage_cols = [
        'diabetes_stage', 'diabetes_risk_score',
        'glucose_fasting', 'glucose_postprandial',
        'insulin_level', 'hba1c'
    ]
    cols_to_drop = [col for col in leakage_cols if col in df_engineered.columns]
    df_engineered = df_engineered.drop(columns=cols_to_drop)

    # 2. Construct advanced clinical derived features
    df_engineered['tc_hdl_ratio'] = df_engineered['cholesterol_total'] / df_engineered['hdl_cholesterol']
    df_engineered['tg_hdl_ratio'] = df_engineered['triglycerides'] / df_engineered['hdl_cholesterol']
    df_engineered['pulse_pressure'] = df_engineered['systolic_bp'] - df_engineered['diastolic_bp']
    df_engineered['map'] = df_engineered['diastolic_bp'] + df_engineered['pulse_pressure'] / 3

    df_engineered['lifestyle_risk_score'] = (
            (df_engineered['smoking_status'] == 'Current').astype(int) +
            (df_engineered['physical_activity_minutes_per_week'] < 150).astype(int) +
            (df_engineered['diet_score'] < 5.0).astype(int) +
            (df_engineered['sleep_hours_per_day'] < 6.0).astype(int)
    )

    # 3. LightGBM ordinal encoding
    edu_map = {'No formal': 0, 'Highschool': 1, 'Graduate': 2, 'Postgraduate': 3}
    income_map = {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}
    smoke_map = {'Never': 0, 'Former': 1, 'Current': 2}

    df_engineered['education_level'] = df_engineered['education_level'].map(edu_map)
    df_engineered['income_level'] = df_engineered['income_level'].map(income_map)
    df_engineered['smoking_status'] = df_engineered['smoking_status'].map(smoke_map)

    # Nominal features (gender, ethnicity, employment_status) are kept as-is
    # and will be cast to 'category' type in the training script

    return df_engineered


def main():
    input_file = 'diabetes_dataset.csv'
    output_file = 'processed_features_for_lgbm.csv'

    print(f"Loading raw data: {input_file} ...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: file '{input_file}' not found. Please ensure it is in the current directory.")
        return

    print("Running LightGBM feature engineering...")
    df_processed = lgbm_feature_engineering(df)

    if 'diagnosed_diabetes' not in df_processed.columns:
        print("Warning: target column 'diagnosed_diabetes' not found.")

    print(f"\nSaving processed data to: {output_file} ...")
    df_processed.to_csv(output_file, index=False)
    print("Preprocessing complete. Data is ready for LightGBM model training.")


if __name__ == "__main__":
    main()
