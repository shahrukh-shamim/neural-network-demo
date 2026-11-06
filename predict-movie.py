import json
import math
import os

# -------------------------------------------------------------------
# 1. LOAD MOVIE DETAILS AND EXISTING WEIGHTS FROM FILES
# -------------------------------------------------------------------
FILE_PATH = "final-weights.json" # Path to the JSON file storing the neural network weights
MOVIE_FILE_PATH = "favourite-movie.json" # Path to the favourite movie


def validate_movie(movie):
    if not isinstance(movie, dict):
        raise ValueError("Expected a JSON object containing movie details.")
    for key in ("movie_title", "movie_length", "imdb_rating"):
        if key not in movie:
            raise ValueError(f"Missing required key: '{key}'.")
    if not isinstance(movie["movie_title"], str) or not movie["movie_title"].strip():
        raise ValueError("'movie_title' must be a nonempty string.")
    for key in ("movie_length", "imdb_rating"):
        value = movie[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"'{key}' must be a finite number.")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"'{key}' must be a finite number within float range.")
    if movie["movie_length"] <= 0:
        raise ValueError("'movie_length' must be greater than zero, in hours.")
    if not 0 <= movie["imdb_rating"] <= 10:
        raise ValueError("'imdb_rating' must be between 0 and 10.")


# Read favourite movie details from file
try:
    with open(MOVIE_FILE_PATH, "r", encoding="utf-8") as f:
        favourite_movie = json.load(f)
    validate_movie(favourite_movie)
    movie_title = favourite_movie["movie_title"]
    x1 = favourite_movie["movie_length"]
    x2 = favourite_movie["imdb_rating"]
    target = 1.0  # Ground truth
except (ValueError, OSError, KeyError) as exc:
    raise SystemExit(
        f"Cannot load '{MOVIE_FILE_PATH}': {exc} "
    ) from None

# Function to validate the structure and values of the weights dictionary
def validate_weights(weights):
    if not isinstance(weights, dict):
        raise ValueError("Expected a JSON object containing weights.")
    for key in ("w1", "w2", "b"):
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



# Prediction requires an existing weights file.
if not os.path.exists(FILE_PATH):
    print(f"'{FILE_PATH}' not found. Please create the file with the required weights.")
    raise SystemExit("File not found. Exiting.")
else:
    print(f"Found existing '{FILE_PATH}', loading weights.")

# Read weights from file
try:
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        saved_weights = json.load(f)
    validate_weights(saved_weights)
except (ValueError, OSError) as exc:
    raise SystemExit(
        f"Cannot load '{FILE_PATH}': {exc} "
    ) from None

w1 = saved_weights["w1"]
w2 = saved_weights["w2"]
b = saved_weights["b"]

summed_input1 = (x1 * w1)
summed_input2 = (x2 * w2)
z = summed_input1 + summed_input2 + b
if not math.isfinite(z):
    raise SystemExit("Cannot predict: the weighted sum exceeds the finite numeric range.")

# Use a nonpositive exponent in either branch to avoid exponential overflow.
if z >= 0:
    prediction = 1.0 / (1.0 + math.exp(-z))
else:
    exp_z = math.exp(z)
    prediction = exp_z / (1.0 + exp_z)

difference = target - prediction
print(f"Movie: {movie_title}")
print(f"Prediction: {prediction:.6f}")
print(f"Difference from target: {difference:.6f}")
if abs(difference) <= 0.2:
    print(f"Prediction is you will like {movie_title} ")
else:
    print(f"Prediction is you will not like {movie_title} ")
