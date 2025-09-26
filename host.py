import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Load model and tokenizer
model = load_model(r"C:\Users\sahil\OneDrive\Desktop\RNN\model\next_word_model.h5")
with open(r"C:\Users\sahil\OneDrive\Desktop\RNN\model\tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 56  # same max_len used during training

st.title("📝 Next Word Prediction ✨")
st.subheader("Type some text and let the model complete your sentence!  ⌨️")

user_input = st.text_input("Type a starting text here ⬇️")

if st.button("Predict ➡️"):
    if user_input.strip() != "":
        text = user_input.strip()

        # Generate next 6 words
        for _ in range(12):  # safer number for coherent sentence
            token_list = tokenizer.texts_to_sequences([text])[0]
            token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
            predicted_probs = model.predict(token_list, verbose=0)
            predicted_index = np.argmax(predicted_probs)

            # Find word from index
            for word, index in tokenizer.word_index.items():
                if index == predicted_index:
                    next_word = word
                    break

            text += " " + next_word  # append predicted word to sentence

        # Display full predicted sentence
        st.write(f"💡 Predicted sentence: **{text}**")
    else:
        st.write("⚠️ Please enter some text to predict!")
