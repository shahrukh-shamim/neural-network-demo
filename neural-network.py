import math

# -------------------------------------------------------------------
# 1. INITIAL SETUP & FEATURES
# -------------------------------------------------------------------
# Feature inputs for Shawshank Redemption:
# x1: Movie Length (2.33 hours), x2: IMDb Rating (9.3)
x1 = 2.33
x2 = 9.3
target = 1.0  # Ground truth: You love this movie (100%)

# Starting weights and bias (matching our exact trace)
w1 = -20.0  # Weight for Movie Length
w2 = 4.0  # Weight for IMDb Rating
b = -1.0  # Bias term
learning_rate = 0.1

print(f"Initial Weights -> Length Weight (w1): {w1}, Rating Weight (w2): {w2}")
print("-" * 65)

# -------------------------------------------------------------------
# 2. FORWARD PASS
# -------------------------------------------------------------------
# Weighted sum (Z)
z = (x1 * w1) + (x2 * w2) + b

# Sigmoid Activation function: 1 / (1 + e^-z)
prediction = 1.0 / (1.0 + math.exp(-z))

print(f"[FORWARD PASS]")
print(f"Weighted Sum (z): {z:.4f}")
print(f"Prediction:       {prediction:.6f} (Target: {target})\n")

# -------------------------------------------------------------------
# 3. BACKPROPAGATION (THE 3 STEPS)
# -------------------------------------------------------------------
# Step 1: Mistake (Direction and Size)
mistake = target - prediction

# Step 2: Activation Sensitivity (Sigmoid derivative at z)
# Formula: prediction * (1 - prediction)
sensitivity = prediction * (1.0 - prediction)

# Step 2 (Cont.): Calculate Blame for each weight
blame_w1 = mistake * x1 * sensitivity
blame_w2 = mistake * x2 * sensitivity

print(f"[BACKPROPAGATION]")
print(f"Step 1 - Mistake:               {mistake:+.6f}")
print(f"Step 2 - Activation Sensitivity: {sensitivity:.8f}")
print(f"Step 2 - Blame for Length (w1):  {blame_w1:.8f}")
print(f"Step 2 - Blame for Rating (w2):  {blame_w2:.8f}\n")

# Step 3: Nudge (Update the Weights)
w1_new = w1 + (learning_rate * blame_w1)
w2_new = w2 + (learning_rate * blame_w2)

print(f"[STEP 3: WEIGHT UPDATES]")
print(f"New Length Weight (w1): {w1_new:.8f} (Shift: {w1_new - w1:+.8f})")
w1 = w1_new
print(f"New Rating Weight (w2): {w2_new:.8f} (Shift: {w2_new - w2:+.8f})")
w2 = w2_new
print("-" * 65)