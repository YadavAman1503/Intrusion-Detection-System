import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    df = pd.read_csv('data/Data.csv')

    if 'Label' not in df.columns:
        print("⚠️ No 'Label' column found. Adding dummy labels.")
        df['Label'] = 0  # Add fake labels to prevent error (for testing only)

    X = df.drop('Label', axis=1)
    y = df['Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save column names to file for later use in prediction
    with open('feature_names.txt', 'w') as f:
        for col in X.columns:
            f.write(f"{col}\n")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, scaler
