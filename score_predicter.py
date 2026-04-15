import numpy as np

X = np.array([1, 2, 3, 4, 5])
Y = np.array([35, 43, 50, 68, 79])


# Initialize parameters
weight = 0.0
bias = 0.0
speed = 0.05
size = float(len(X))

def train(epochs):
    global weight, bias, speed, size
    w = weight
    b = bias
    learning_speed = speed
    n = size

    print(f"Begin training with inital weight: {w}, bias:{b}\n")

    for i in range(epochs):
        # Make predictions
        y_predictions = (w * X) + b


        # Calculate loss
        loss = (sum((Y - y_predictions)**2)/n)

        # Calculate Gradients
        db = (-2/n) * sum(Y-y_predictions)
        dw = (-2/n) * sum(X * (Y-y_predictions))


        w = w - (learning_speed * dw)
        b = b - (learning_speed * db)

        if i % 10 == 0:
            print(f"Epoch {i}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")
        
    print(f"\nTraining Complete!")
    print(f"Final Model: {w:.2f} * Hours + {b:.2f}")

    weight = w
    bias = b
    speed = learning_speed
    size = n

def predict(new_hour):
    return [new_hour, (weight * new_hour + bias)]


if __name__ == "__main__":
    train(200)

    prediction = predict(8)


    print(f"Predicted score for {prediction[0]} hours of study: {prediction[1]:.2f}")