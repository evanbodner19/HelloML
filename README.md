# HelloML

A minimal linear regression model built from scratch using NumPy. Predicts a score based on hours of study, trained via gradient descent.

## How it works

- Fits a line `score = weight * hours + bias` to training data
- Uses mean squared error loss with gradient descent to optimize `weight` and `bias`
- Prints loss every 10 epochs during training

## Requirements

- Python 3.x
- NumPy

```bash
pip install numpy
```

## Usage

```bash
python score_predicter.py
```
