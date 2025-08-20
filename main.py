from src.train_model import train
from src.predict import predict

def main():
    train()

    print("✅ Training complete. Now predicting a sample...")

    # Create a test input with 76 features, all set to 0
    with open('feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f]

    sample_input = {feature: 0 for feature in feature_names}
    result = predict(sample_input)
    
    print(f"🚨 Prediction result: {'Attack' if result == 1 else 'Normal'}")

if __name__ == "__main__":
    main()
