import streamlit as st
import pickle
import numpy as np
from dotenv import load_dotenv, find_dotenv
from googleapiclient.discovery import build
import google.generativeai as genai

# Configure the Generative AI library with an API key
GOOGLE_API_KEY = "Replace with your api"
genai.configure(api_key=GOOGLE_API_KEY)

# YouTube Data API Key
YOUTUBE_API_KEY = "Replace with your api"

# Load the prediction model
try:
    with open("xgb_model_final.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("The model file 'xgb_model_final.pkl' is not found. Please ensure it is in the correct directory.")
    st.stop()

st.title("Thyroid Prediction, Diet Plan, and Exercise Recommendations")

# Initialize session state for prediction result
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# Input fields for all columns
age = st.number_input("Enter Age", value=0, format="%d")
sex = st.selectbox("Select Sex", ["Male", "Female"])
on_thyroxine = st.selectbox("Are you on thyroxine?", ["No", "Yes"])
on_antithyroid_meds = st.selectbox("Are you on antithyroid medications?", ["No", "Yes"])
sick = st.selectbox("Are you sick?", ["No", "Yes"])
pregnant = st.selectbox("Are you pregnant?", ["No", "Yes"])
thyroid_surgery = st.selectbox("Have you undergone thyroid surgery?", ["No", "Yes"])
I131_treatment = st.selectbox("Have you received I131 treatment?", ["No", "Yes"])
lithium = st.selectbox("Are you taking lithium?", ["No", "Yes"])
goitre = st.selectbox("Do you have goitre?", ["No", "Yes"])
tumor = st.selectbox("Do you have a tumor?", ["No", "Yes"])
hypopituitary = st.selectbox("Do you have hypopituitary condition?", ["No", "Yes"])
psych = st.selectbox("Do you have any psychological condition?", ["No", "Yes"])
TSH = st.number_input("Enter TSH level", value=0.0, format="%.2f")
T3 = st.number_input("Enter T3 level", value=0.0, format="%.2f")
TT4 = st.number_input("Enter TT4 level", value=0.0, format="%.2f")
T4U = st.number_input("Enter T4U level", value=0.0, format="%.2f")
FTI = st.number_input("Enter FTI level", value=0.0, format="%.2f")

# Convert categorical fields to numerical
sex_encoded = 1 if sex == "Female" else 0
on_thyroxine_encoded = 1 if on_thyroxine == "Yes" else 0
on_antithyroid_meds_encoded = 1 if on_antithyroid_meds == "Yes" else 0
sick_encoded = 1 if sick == "Yes" else 0
pregnant_encoded = 1 if pregnant == "Yes" else 0
thyroid_surgery_encoded = 1 if thyroid_surgery == "Yes" else 0
I131_treatment_encoded = 1 if I131_treatment == "Yes" else 0
lithium_encoded = 1 if lithium == "Yes" else 0
goitre_encoded = 1 if goitre == "Yes" else 0
tumor_encoded = 1 if tumor == "Yes" else 0
hypopituitary_encoded = 1 if hypopituitary == "Yes" else 0
psych_encoded = 1 if psych == "Yes" else 0

# Prediction button
if st.button("Predict Thyroid Condition"):
    try:
        # Prepare the input data
        input_data = np.array([[age, sex_encoded, on_thyroxine_encoded, on_antithyroid_meds_encoded,
                                sick_encoded, pregnant_encoded, thyroid_surgery_encoded, I131_treatment_encoded,
                                lithium_encoded, goitre_encoded, tumor_encoded, hypopituitary_encoded,
                                psych_encoded, TSH, T3, TT4, T4U, FTI]])

        # Predict
        st.session_state.prediction_result = model.predict(input_data)[0]

        # Display result
        if st.session_state.prediction_result == 1:
            st.header("Predicted Condition: Hypothyroid")
        elif st.session_state.prediction_result == 2:
            st.header("Predicted Condition: Hyperthyroid")
        else:
            st.header("Predicted Condition: No Thyroid")
            st.subheader("Congratulations! You do not have any thyroid condition.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Show options based on prediction
if st.session_state.prediction_result in [1, 2]:  # Hypothyroid or Hyperthyroid
    # Generate diet plan button
    if st.button("Generate Diet Plan"):
        try:
            # Generate diet plan using Generative AI
            input_prompt = f"""
            You are a nutritionist. Provide a personalized diet plan for a patient based on the following details:
            Age: {age}, Sex: {sex}, TSH Level: {TSH}, T3 Level: {T3}, TT4 Level: {TT4},
            T4U Level: {T4U}, FTI Level: {FTI}, Thyroid Condition: {'Hypothyroid' if st.session_state.prediction_result == 1 else 'Hyperthyroid'}.
            Include meals for each day of the week (Monday to Sunday) with specific recommendations for Breakfast, Lunch, Snacks, and Dinner.
            """
            with st.spinner("Generating diet plan..."):
                model = genai.GenerativeModel("gemini-1.5-pro-latest")
                response = model.generate_content([input_prompt])
                st.subheader("Personalized Diet Plan")
                st.write(response.text)
        except Exception as e:
            st.error(f"An error occurred while generating the diet plan: {e}")

    # Exercise videos button
    if st.button("Show Exercise Videos"):
        try:
            condition = "Hypothyroid" if st.session_state.prediction_result == 1 else "Hyperthyroid"
            search_query = f"{condition} exercise videos"

            # Fetch top YouTube video links
            youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
            request = youtube.search().list(
                q=search_query,
                part="snippet",
                maxResults=5,
                type="video"
            )
            response = request.execute()

            st.subheader(f"Top 5 Exercise Videos for {condition}")
            for item in response["items"]:
                video_title = item["snippet"]["title"]
                video_url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                st.write(f"[{video_title}]({video_url})")
        except Exception as e:
            st.error(f"An error occurred while fetching exercise videos: {e}")
