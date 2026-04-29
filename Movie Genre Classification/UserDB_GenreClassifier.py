import json
import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

DB_FILE = "users.json"

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

def predict_genre(description):
    cleaned = clean_text(description)
    vector = vectorizer.transform([cleaned]).toarray()
    return model.predict(vector)[0]

def load_users():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_users(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

def register_user(name, email, mobile, age, genres):
    users = load_users()

    users[email] = {
        "name": name,
        "mobile": mobile,
        "age": age,
        "preferred_genres": genres,
        "prediction_history": []
    }

    save_users(users)
    print("User registered successfully!")

def add_prediction_history(email, description):
    users = load_users()

    if email not in users:
        print("User not found!")
        return

    predicted_genre = predict_genre(description)

    users[email]["prediction_history"].append({
        "description": description,
        "predicted_genre": predicted_genre
    })

    save_users(users)
    print("Prediction saved!")

def get_user(email):
    users = load_users()
    return users.get(email, "User not found")

def get_user_history(email):
    users = load_users()

    if email not in users:
        return "User not found"

    return users[email]["prediction_history"]

def update_user_genres(email, new_genres):
    users = load_users()

    if email not in users:
        print("User not found!")
        return

    users[email]["preferred_genres"] = new_genres
    save_users(users)

    print("Preferences updated!")

def delete_user(email):
    users = load_users()

    if email in users:
        del users[email]
        save_users(users)
        print("User deleted!")
    else:
        print("User not found!")

if __name__ == "__main__":
    
    register_user(
        name="Ankit",
        email="ankit@gmail.com",
        mobile="1234567890",
        age=19,
        genres=["drama", "action"]
    )

    add_prediction_history(
        "ankit@gmail.com",
        "A haunted house with ghosts and dark secrets"
    )

    add_prediction_history(
        "ankit@gmail.com",
        "A romantic love story between two strangers"
    )

    print("\nUser Data:")
    print(get_user("ankit@gmail.com"))

    print("\nPrediction History:")
    print(get_user_history("ankit@gmail.com"))

    update_user_genres("ankit@gmail.com", ["horror", "thriller"])

    # Delete User
    # delete_user("ankit@gmail.com")
