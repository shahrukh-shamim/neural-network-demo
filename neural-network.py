import math

# -------------------------------------------------------------------
# 1. INITIAL SETUP & FEATURES
# -------------------------------------------------------------------
x1 = 2.33  # Movie Length
x2 = 9.3  # IMDb Rating
target = 1.0  # Ground truth

w1 = -20.0  # Weight for Length
w2 = 4.0  # Weight for Rating
b = -1.0  # Bias
learning_rate = 0.1

iteration = 0

print(f"Initial Weights -> w1: {w1}, w2: {w2}")
print("-" * 65)

while True:
    iteration += 1

    # Forward Pass
    z = (x1 * w1) + (x2 * w2) + b
    prediction = 1.0 / (1.0 + math.exp(-z))

    # Step 1: Mistake
    mistake = target - prediction

    # Check termination threshold
    if abs(mistake) <= 0.001:
        print(f"\n[TARGET REACHED on Iteration {iteration}!]")
        print(f"Final Weighted Sum (z): {z:.4f}")
        print(f"Final Prediction:       {prediction:.6f}")
        print(f"Final Weights:          w1 = {w1:.4f}, w2 = {w2:.4f}")
        break

    # Step 2: Sensitivity & Blame
    sensitivity = prediction * (1.0 - prediction)
    blame_w1 = mistake * x1 * sensitivity
    blame_w2 = mistake * x2 * sensitivity

    # Step 3: Update Weights
    w1 += learning_rate * blame_w1
    w2 += learning_rate * blame_w2

    # Print progress snapshot every 5,000 iterations
    if iteration % 5000 == 0:
        print(
            f"Iter {iteration:6d} | z: {z:8.4f} | Prediction: {prediction:.6f} | w1: {w1:8.4f} | w2: {w2:8.4f}"
        )