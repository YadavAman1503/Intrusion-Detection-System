import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from src.data_preprocessing import preprocess_data

def train():
    print("🔁 Training the Intrusion Detection Model...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data()

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model and scaler
    with open('ids_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("✅ Model and scaler saved.")
