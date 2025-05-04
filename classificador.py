import tensorflow as tf
import numpy as np
import cv2
import os
import google.generativeai as genai
import webbrowser

# === CONFIGURATIONS ===
API_KEY_GEMINI = "AIzaSyAn3Z485B1p7iBESKyG_kXv3qcoQwoG7EI"
MODEL_PATH = "balanced_mobilenet_model.h5"
SPECTROGRAMS_PATH = r"C:\Users\Administrator\Documents\my_ecoauralia_project\spectrograms"

# === AUTHENTICATE GEMINI ===
genai.configure(api_key=API_KEY_GEMINI)

# === INITIALIZE MODELS ===
text_model = genai.GenerativeModel("models/gemini-1.5-pro")
cnn_model = tf.keras.models.load_model(MODEL_PATH)
labels = sorted(os.listdir(SPECTROGRAMS_PATH))

# === FUNCTION: CLASSIFY SPECTROGRAM ===
def classify_spectrogram(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Image not found: {image_path}")
        return None
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = cnn_model.predict(img)
    top_indices = np.argsort(pred[0])[::-1][:3]  # top-3 predictions

    print("\n🔊 Model predictions (spectrogram):")
    for i, idx in enumerate(top_indices):
        species = labels[idx]
        confidence = pred[0][idx]
        marker = "✅" if i == 0 else "→"
        print(f"{marker} {i+1}. {species} ({confidence:.1%})")

    if pred[0][top_indices[0]] < 0.6:
        print("⚠️ Prediction with low confidence. There may be doubt between species.")

    return labels[top_indices[0]]

# generate text with gemini focused on animal description
def generate_description(animal):
    prompt = f"Describe the animal '{animal}' in detail. Where it lives, what sound it makes, and interesting facts about it."
    response = text_model.generate_content(prompt)
    print("\n📖 Description:")
    print(response.text)
    return response.text

# search for real images of the animal
def search_real_image(animal):
    url = f"https://www.google.com/search?tbm=isch&q={animal}"
    print(f"\n🔎 Opening real images of: {animal}")
    webbrowser.open(url)
