# text_emotion_detection


#1. Dataset Used
The code doesn’t explicitly load or describe the dataset, but since the model file text_emotion.pkl is loaded, the dataset was likely used offline to train that model.

Given the emotion classes in emotions_emoji_dict (anger, disgust, fear, happy, joy, neutral, sad, sadness, shame, surprise), the dataset probably contains text samples labeled with these emotions.

Typical datasets for emotion detection include:

Emotion dataset from NLP libraries

#2.Approach Summary
The app uses a pretrained model loaded with joblib (text_emotion.pkl), likely a classical ML model or pipeline (e.g., TF-IDF + Logistic Regression or similar).

The app:

Takes raw text input via a Streamlit text area.

Uses pipe_lr.predict() to predict the most likely emotion class.

Uses pipe_lr.predict_proba() to get probabilities for all emotion classes.

Displays the predicted emotion with an emoji and confidence score.

Shows a bar chart of prediction probabilities using Altair.

No deep learning model or preprocessing steps are shown here, so these must be baked into the loaded pipeline (pipe_lr).

#3. Dependencies
streamlit
pandas
numpy
altair
joblib
scikit-learn 
