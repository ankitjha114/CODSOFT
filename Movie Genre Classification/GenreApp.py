import streamlit as st
from GenreAPI import predict_and_recommend
from UserDB_GenreClassifier import register_user, add_prediction_history

st.set_page_config(page_title="CineMatch Genre AI", layout="wide")


st.markdown("""
<style>
    .stApp {
        background-color: black;
        color: white;
    }

    h1, h2, h3 {
        color: #E50914;
        text-align: center;
    }

    label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }

    input, textarea {
        background-color: #1f1f1f !important;
        color: white !important;
        border: 1px solid #333 !important;
        border-radius: 6px !important;
    }

    .stSelectbox div[data-baseweb="select"] {
        background-color: #1f1f1f !important;
        color: white !important;
        border-radius: 6px !important;
    }

    .stNumberInput input {
        background-color: #1f1f1f !important;
        color: white !important;
    }

    .stButton>button {
        background-color: #E50914;
        color: white;
        border-radius: 8px;
        height: 45px;
        width: 200px;
        font-weight: bold;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div style="text-align:center;">
    <h1>Movie Genre Classifier</h1>
    <p style="color:gray;">Predict Genre + Get Smart Recommendations</p>
</div>
""", unsafe_allow_html=True)


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


if not st.session_state.logged_in:

    st.title("Create Your Account")

    with st.form("login_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        mobile = st.text_input("Mobile Number")
        age = st.number_input("Age", min_value=10, max_value=100)
        genres = st.multiselect(
            "Favorite Genres",
            ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller", "Horror"]
        )

        submit = st.form_submit_button("Create Account")

        if submit:
            register_user(name, email, mobile, age, genres)

            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.session_state.user_genres = genres

            st.success("Login Successful")
            st.rerun()


else:

    st.subheader("Enter Movie Description")

    user_input = st.text_area("Type movie plot or description here...")

    if st.button("Predict Genre & Recommend"):

        if user_input.strip() == "":
            st.warning("Please enter description")
        else:
            result = predict_and_recommend(user_input)

            add_prediction_history(st.session_state.user_email, user_input)


            st.success(f"Predicted Genre: {result['predicted_genre']}")


            st.subheader("Recommended Movies")

            movies = result["recommended_movies"]
            posters = result["posters"]

            if movies:
                cols = st.columns(5)
                for i in range(len(movies)):
                    with cols[i]:
                        st.image(posters[i])
                        st.markdown(f"**{movies[i]}**")
            else:
                st.info("No recommendations found for this genre.")


    st.subheader("Your Prediction History")

    from UserDB_GenreClassifier import get_user_history
    history = get_user_history(st.session_state.user_email)

    if isinstance(history, list) and len(history) > 0:
        for item in history[::-1][:5]:
            st.markdown(f"""
            - **Genre:** {item['predicted_genre']}  
              **Text:** {item['description'][:100]}...
            """)
    else:
        st.info("No history yet.")