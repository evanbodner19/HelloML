import numpy as np
import pandas as pd

training_data = pd.read_csv("training_data.csv")

X1 = training_data['Hours Studied'].values
X2 = training_data['Practice Tests Taken'].values
X3 = training_data['Classes Missed'].values
X4 = training_data['Sleep Hours'].values
X5 = training_data['Previous Scores'].values
Y = training_data['Final Score'].values

# Stack all features into a matrix (n_samples x n_features)
X_raw = np.column_stack([X1, X2, X3, X4, X5])
FEATURE_NAMES = ['Hours Studied', 'Practice Tests Taken', 'Classes Missed', 'Sleep Hours', 'Previous Scores']

# Normalize features to zero mean and unit variance
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X = (X_raw - X_mean) / X_std

# Initialize parameters
weights = np.zeros(X.shape[1])
bias = 0.0
speed = 0.02

def train(epochs):
    global weights, bias
    w = weights.copy()
    b = bias
    n = float(len(Y))

    print(f"Begin training with initial weights: {w}, bias: {b}\n")

    for i in range(epochs):
        # Make predictions
        y_predictions = X @ w + b

        # Calculate loss (MSE)
        loss = np.sum((Y - y_predictions) ** 2) / n

        # Calculate gradients
        error = Y - y_predictions
        dw = (-2 / n) * (X.T @ error)
        db = (-2 / n) * np.sum(error)

        w = w - (speed * dw)
        b = b - (speed * db)

        if i % 10 == 0:
            print(f"Epoch {i}: Loss = {loss:.4f}, b = {b:.4f}")

    print(f"\nTraining Complete!")
    for name, coef in zip(FEATURE_NAMES, w):
        print(f"  {name}: {coef:.4f}")
    print(f"  Bias: {b:.4f}")

    weights = w
    bias = b

def predict(hours_studied, practice_tests, classes_missed, sleep_hours, previous_scores):
    features = np.array([hours_studied, practice_tests, classes_missed, sleep_hours, previous_scores])
    features_normalized = (features - X_mean) / X_std
    return weights @ features_normalized + bias


if __name__ == "__main__":
    train(500)

    score = predict(
        hours_studied=8,
        practice_tests=3,
        classes_missed=1,
        sleep_hours=7,
        previous_scores=85
    )
    print(f"\nPredicted score: {score:.2f}")
