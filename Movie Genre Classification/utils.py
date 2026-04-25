import numpy as np
import pandas as pd
import re
import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

ps = PorterStemmer()


def load_train_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                id_, title, genre, desc = parts
                data.append([id_, title, genre, desc])
    return pd.DataFrame(data, columns=['id', 'title', 'genre', 'description'])


def load_test_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 3:
                id_, title, desc = parts
                data.append([id_, title, desc])
    return pd.DataFrame(data, columns=['id', 'title', 'description'])


def load_solution(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                id_, title, genre, desc = parts
                data.append([id_, genre])
    return pd.DataFrame(data, columns=['id', 'genre'])


train_df = load_train_data("train_data.txt")
test_df = load_test_data("test_data.txt")
solution_df = load_solution("test_data_solution.txt")

print(train_df.head())
print(train_df.shape)
print(test_df.shape)


train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

print(train_df.isnull().sum())
print(train_df.duplicated().sum())


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

train_df['clean_desc'] = train_df['description'].apply(clean_text)
test_df['clean_desc'] = test_df['description'].apply(clean_text)

print(train_df.head())


vectorizer = TfidfVectorizer(max_features=10000)

X = vectorizer.fit_transform(train_df['clean_desc']).toarray()
y = train_df['genre']

X_test = vectorizer.transform(test_df['clean_desc']).toarray()

print(X.shape)
print(X[0])


models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

predictions = {}
accuracies = {}

print("\nTraining Models...\n")

for name, model in models.items():
    model.fit(X, y)
    preds = model.predict(X_test)
    predictions[name] = preds
    acc = accuracy_score(solution_df['genre'], preds)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc}")


plt.figure()
plt.bar(accuracies.keys(), accuracies.values())
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()


best_model = models["SVM"]
best_preds = predictions["SVM"]


cm = confusion_matrix(solution_df['genre'], best_preds, labels=best_model.classes_)

plt.figure()
sns.heatmap(cm, annot=False, fmt='d')
plt.title("Confusion Matrix (SVM)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


print("\nClassification Report:\n")
print(classification_report(solution_df['genre'], best_preds))


def predict_genre(text):
    text = clean_text(text)
    vec = vectorizer.transform([text]).toarray()
    return best_model.predict(vec)[0]

print("\nSample Prediction:")
print(predict_genre("A haunted house with ghosts and mystery"))


pickle.dump(best_model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model and vectorizer saved!")


output_df = pd.DataFrame({
    "id": test_df['id'],
    "predicted_genre": best_preds
})

output_df.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")