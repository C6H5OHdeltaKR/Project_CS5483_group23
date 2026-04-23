import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


def generate_advanced_features(df):
    """
    Construct advanced features and drop all target-leakage columns
    to satisfy the early-risk prediction objective.
    """
    df_engineered = df.copy()

    # 1. Drop leakage columns and direct biochemical diagnostic indicators
    leakage_cols = [
        'diabetes_stage', 'diabetes_risk_score',
        'glucose_fasting', 'glucose_postprandial',
        'insulin_level', 'hba1c'
    ]
    df_engineered = df_engineered.drop(columns=[col for col in leakage_cols if col in df_engineered.columns])

    # 2. Construct advanced clinical features (non-glucose early-risk indicators)

    # a. Lipid metabolism (dyslipidemia often precedes hyperglycemia)
    df_engineered['tc_hdl_ratio'] = df_engineered['cholesterol_total'] / df_engineered['hdl_cholesterol']
    df_engineered['tg_hdl_ratio'] = df_engineered['triglycerides'] / df_engineered['hdl_cholesterol']

    # b. Blood pressure derived features
    df_engineered['pulse_pressure'] = df_engineered['systolic_bp'] - df_engineered['diastolic_bp']
    df_engineered['map'] = df_engineered['diastolic_bp'] + df_engineered['pulse_pressure'] / 3

    # c. Composite lifestyle risk score (0-4)
    df_engineered['lifestyle_risk_score'] = (
            (df_engineered['smoking_status'] == 'Current').astype(int) +
            (df_engineered['physical_activity_minutes_per_week'] < 150).astype(int) +
            (df_engineered['diet_score'] < 5.0).astype(int) +
            (df_engineered['sleep_hours_per_day'] < 6.0).astype(int)
    )

    return df_engineered


def main():
    input_file = 'diabetes_dataset.csv'
    output_file = 'processed_diabetes_data_early_risk.csv'

    print(f"Loading data: {input_file} ...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: file '{input_file}' not found.")
        return

    print("Dropping leakage columns and running feature engineering...")
    df_featured = generate_advanced_features(df)

    target_col = 'diagnosed_diabetes'
    if target_col not in df_featured.columns:
        print(f"Error: target column '{target_col}' not found in data.")
        return

    X = df_featured.drop(columns=[target_col])
    y = df_featured[target_col]

    # --- Define categorical encoding strategy ---
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
            ('num', 'passthrough', numeric_features)
        ]
    )

    preprocessor.set_output(transform="pandas")

    print("Encoding categorical features...")
    X_processed = preprocessor.fit_transform(X)

    df_final = pd.concat([X_processed, y.reset_index(drop=True)], axis=1)

    df_final.columns = [col.split('__')[-1] if '__' in col else col for col in df_final.columns]

    print(f"Saving early-risk prediction data to: {output_file} ...")
    df_final.to_csv(output_file, index=False)
    print("Preprocessing complete. Use the new CSV file to run the cross-validation script.")


if __name__ == "__main__":
    main()
