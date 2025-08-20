import pickle
import numpy as np

def predict(sample_dict):
    with open('ids_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Read the order of features
    with open('feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f]

    # Build sample input in correct order
    sample_values = [sample_dict.get(name, 0) for name in feature_names]
    sample_values = np.array(sample_values).reshape(1, -1)

    sample_scaled = scaler.transform(sample_values)
    prediction = model.predict(sample_scaled)
    return prediction[0]
