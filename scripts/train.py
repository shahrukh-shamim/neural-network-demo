from pathlib import Path

import json
import math
import os
import tempfile
import time

# -------------------------------------------------------------------
# 1. LOAD WEIGHTS FROM FILE (OR INITIALIZE IF FILE DOESN'T EXIST)
# -------------------------------------------------------------------
FILE_PATH = Path(__file__).resolve().parents[1] / "data" / "weights.json" # Path to the JSON file storing the neural network weights

# Function to validate the structure and values of the weights dictionary
def validate_weights(weights):
    if not isinstance(weights, dict):
        raise ValueError("Expected a JSON object containing weights and learning_rate.")
    for key in ("w1", "w2", "b", "learning_rate"):
        if key not in weights:
            raise ValueError(f"Missing required key: '{key}'.")
        value = weights[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"'{key}' must be a finite number.")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"'{key}' must be a finite number within float range.")
    if weights["learning_rate"] <= 0:
        raise ValueError("'learning_rate' must be greater than zero.")

# Function to save the weights dictionary to a JSON file safely
def save_weights(path, weights):
    validate_weights(weights) # Ensure the weights are valid before saving
    directory = os.path.dirname(os.path.abspath(path)) # Get the directory of the target file to place the temporary file on the same filesystem
    temporary_path = None # Initialize the temporary file path variable
    try:
        # Keep the temporary file on the same filesystem as the destination.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory,
            prefix=".weights-", suffix=".tmp", delete=False
        ) as f:
            temporary_path = f.name
            json.dump(weights, f, indent=4, allow_nan=False)
            f.flush() # Flush the internal buffer to the OS
            os.fsync(f.fileno()) # Ensure the data is physically written to disk
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.remove(temporary_path)


# Check if file exists; if not, create it with initial default weights
if not os.path.exists(FILE_PATH):
    default_weights = {"w1": -20.0, "w2": 4.0, "b": -1.0, "learning_rate": 0.01} # Initial default weights for the neural network
    save_weights(FILE_PATH, default_weights)
    print(f"Created new '{FILE_PATH}' with default weights.")
else:
    print(f"Found existing '{FILE_PATH}', loading weights.")

time.sleep(5)

# Read initial weights from file
try:
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        saved_weights = json.load(f)
    validate_weights(saved_weights)
except (ValueError, OSError) as exc:
    raise SystemExit(
        f"Cannot load '{FILE_PATH}': {exc} "
        "Training stopped; the existing file has not been overwritten. "
        "Correct the file, or rename it to create fresh defaults on the next run."
    ) from None

w1 = saved_weights["w1"]
w2 = saved_weights["w2"]
b = saved_weights["b"]
learning_rate = saved_weights["learning_rate"]

# print(
#     f"Loaded Initial Weights from file -> w1: {w1}, w2: {w2}, b: {b}, learning_rate: {learning_rate}"
# )
# time.sleep(2) # Pause for 2 seconds before starting the training loop

# -------------------------------------------------------------------
# 2. TRAINING SETUP
# -------------------------------------------------------------------
x1 = 2.33  # Movie Length
x2 = 9.3  # IMDb Rating
target = 1.0  # Ground truth

# learning_rate = 0.1
max_iterations = 100_000
iteration = 0

while iteration <= max_iterations:

    print("-" * 65)
    print(f"Iteration {iteration} | Current Weights: w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}, learning_rate = {learning_rate:.4f}")
    print("-" * 65)
    print("\n\n")
    # Forward Pass
    z = (x1 * w1) + (x2 * w2) + b
    prediction = 1.0 / (1.0 + math.exp(-z))

    # Step 1: Mistake
    mistake = target - prediction

    # Check termination threshold
    if abs(mistake) <= 0.001:
        print(f"\n[TARGET REACHED after {iteration} weight updates!]")
        print(f"Final Weights: w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}, learning_rate = {learning_rate:.4f}")
        break

    if iteration == max_iterations:
        print(f"\n[ITERATION LIMIT REACHED: {max_iterations} weight updates]")
        break

    # Step 2: Sensitivity & Blame
    sensitivity = prediction * (1.0 - prediction)
    blame_w1 = mistake * x1 * sensitivity
    blame_w2 = mistake * x2 * sensitivity

    # Step 3: Update Weights in memory
    w1 += learning_rate * blame_w1
    w2 += learning_rate * blame_w2
    b += learning_rate * mistake * sensitivity

    # Write updated weights back to file after each iteration
    current_weights = {"w1": w1, "w2": w2, "b": b, "learning_rate": learning_rate}
    save_weights(FILE_PATH, current_weights)

    if iteration > 0 and iteration % 5000 == 0:
        print(
            f"Iter {iteration:6d} | z: {z:8.4f} | Prediction: {prediction:.6f} | File updated!"
        )

    iteration += 1

