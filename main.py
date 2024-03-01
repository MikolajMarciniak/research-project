import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

# Specify the path to the JSON file
file_path = "./AMAZON_FASHION_5.json"

# Load the data from the JSON file
data = []
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        entry = json.loads(line)
        if "summary" in entry:
            data.append(entry)

# Extract text summaries and overall ratings
summaries = [d["summary"] for d in data]
ratings = [d["overall"] for d in data]

# Preprocess text data
# For simplicity, we'll just use the text as is
processed_summaries = summaries

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(processed_summaries)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, ratings, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot learning curve
train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure()
plt.title("Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Mean Squared Error")
plt.grid()

plt.plot(train_sizes, train_scores_mean, label="Training Error")
plt.plot(train_sizes, test_scores_mean, label="Cross-validation Error")
plt.legend(loc="best")

# Save plot as JPEG
plt.savefig("learning_curve.jpg")


# Output predictions
for i in range(len(y_test)):
    print(f"Actual: {y_test[i]}, Predicted: {y_pred_rounded[i]}, {y_pred[i]}")

