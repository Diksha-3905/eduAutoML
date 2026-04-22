import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def prepare_data(file_path, target_column, drop_threshold=0.5):
    df = pd.read_csv(file_path)

    # Drop rows with missing target
    df = df.dropna(subset=[target_column])

    # Warn if any column has too many missing values
    missing_ratios = df.isnull().mean()
    high_missing = missing_ratios[missing_ratios > drop_threshold]
    if not high_missing.empty:
        print("⚠️ Warning: Columns with high missing values:")
        print(high_missing)

    # Drop high-missing columns
    df = df.drop(columns=high_missing.index)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical features (one-hot encode)
    X = pd.get_dummies(X, drop_first=True)

    # Impute missing values for numeric columns
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Encode target if categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y
