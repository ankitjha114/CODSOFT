# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# ps = PorterStemmer()


# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))

# st.title("Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message")

# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# # Download required NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# # Load model
# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))

# # UI
# st.title("📩 Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):

#     if input_sms.strip() == "":
#         st.warning("Please enter a message!")
#     else:
#         # 1. preprocess
#         transformed_sms = transform_text(input_sms)

#         # 2. vectorize
#         vector_input = tfidf.transform([transformed_sms])

#         # 3. predict
#         result = model.predict(vector_input)[0]

#         # 4. Display
#         if result == 1:
#             st.error(" Spam Message")
#         else:
#             st.success(" Not Spam")






# import streamlit as st
# import pickle
# import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# # Download required data (only runs once)
# nltk.download('punkt')
# nltk.download('stopwords')

# ps = PorterStemmer()

# # Text preprocessing function
# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# # Load model and vectorizer
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

# # UI
# st.set_page_config(page_title="Spam Classifier", layout="centered")

# st.title("📩 Email/SMS Spam Classifier")

# st.write("Enter a message below to check whether it is spam or not.")

# # ✅ IMPORTANT: unique key added
# input_sms = st.text_area("Enter the message", key="sms_input")

# # Button with unique key
# if st.button('Predict', key="predict_btn"):

#     if input_sms.strip() == "":
#         st.warning("⚠️ Please enter a message")
#     else:
#         # Preprocess
#         transformed_sms = transform_text(input_sms)

#         # Vectorize
#         vector_input = tfidf.transform([transformed_sms])

#         # Predict
#         result = model.predict(vector_input)[0]

#         # Output
#         if result == 1:
#             st.error("🚨 Spam Message")
#         else:
#             st.success("✅ Not Spam")




import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load trained model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# UI
st.set_page_config(page_title="Spam Classifier")
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message", key="sms_input")

if st.button('Predict', key="predict_btn"):

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])

        try:
            result = model.predict(vector_input)[0]

            if result == 1:
                st.error("🚨 Spam Message")
            else:
                st.success("✅ Not Spam")

        except Exception as e:
            st.error("Model not trained properly. Please retrain and save again.")