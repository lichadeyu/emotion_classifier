#========================import packages=========================================================
import streamlit as st
import numpy as np
import re
import pickle
import nltk
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

#========================loading the save files==================================================
# Load LSTM model
model = load_model('model1.h5')

# with open('tokenizer.pkl', 'rb') as handle:
#     tokenizer = pickle.load(handle)

#========================repeating the same functions==========================================
def sentence_cleaning(sentence):
    stemmer = PorterStemmer()
    corpus = []
    text = re.sub("[^a-zA-Z]", " ", sentence)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    text = " ".join(text)
    corpus.append(text)
    one_hot_word = [one_hot(word, 11000) for word in corpus]
    pad = pad_sequences(sequences=one_hot_word, maxlen=300, padding='pre')
    return pad

def predict_emotion(input_text):
    cleaned_text = sentence_cleaning(input_text)
    
    # Predict emotion
    predicted_probabilities = model.predict(cleaned_text)
    predicted_label = np.argmax(predicted_probabilities)
    
    # Mapping labels
    label_mapping = {0: 'Joy', 1: 'Fear', 2: 'Anger', 3: 'Love', 4: 'Sadness', 5: 'Surprise'}
    predicted_emotion = label_mapping[predicted_label]
    
    probability = np.max(predicted_probabilities)
    
    return predicted_emotion, probability

#==================================creating app====================================
# App
st.title("Six Human Emotions Detection App")
st.write("=================================================")
st.write("['Joy','Fear','Anger','Love','Sadness','Surprise']")
st.write("=================================================")

# taking input from user
user_input = st.text_input("Enter your text here:")

if st.button("Predict"):
    predicted_emotion, probability = predict_emotion(user_input)
    st.write("Predicted Emotion:", predicted_emotion)
    st.write("Probability:", probability)
