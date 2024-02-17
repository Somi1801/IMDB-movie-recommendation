import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Read the dataset containing movie features
data = pd.read_csv('movie_metadata (1).csv')

# Separate the data into features and labels
X = data[['Released_year', 'director_name', 'genres', 'imdb_score']]
y = data['movie_title']

# Create a CountVectorizer to convert text to numerical features
vectorizer = CountVectorizer()
X_transformed = vectorizer.fit_transform(X.astype(str).apply(' '.join, axis=1))

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_transformed, y)

# Create a tkinter window
window = tk.Tk()
window.title("Movie Recommendation System")

# Entry fields for movie features (avoid global variables)
entries = {}

# Function to recommend movies based on user input
def recommend_movies():
    features = {entry_name: entry.get() for entry_name, entry in entries.items()}

    # Convert the user input into a DataFrame
    input_df = pd.DataFrame([features])

    # Transform the input using the same vectorizer used for training
    input_transformed = vectorizer.transform(input_df.astype(str).apply(' '.join, axis=1))

    # Predict top 5 recommended movies based on user input
    recommended_movies = model.predict(input_transformed)
    messagebox.showinfo("Movie Recommendation", f"Recommended Movies: {', '.join(recommended_movies[:5])}")

# Entry fields for movie features
labels = ["Released Year", "director", "Genre", "IMDB Rating"]

for label in labels:
    label_widget = tk.Label(window, text=f"Enter {label}:")
    label_widget.pack()

    entry = tk.Entry(window)
    entry.pack()
    entries[label] = entry  # Store entries in a dictionary for easy access

# Button to get movie recommendations
recommend_button = tk.Button(window, text="Get Recommendations", command=recommend_movies)
recommend_button.pack()

window.mainloop()
