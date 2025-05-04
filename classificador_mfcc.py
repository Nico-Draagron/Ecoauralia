"""
mfcc_classifier.py

Script used in the EcoAuralia project to identify species of farm animals (initially).
This classifier uses the MFCC model, utilizing the same .ogg files and being trained by the "train_mfcc_model.py" file.
"""

import librosa
import numpy as np
import tensorflow as tf
import os

# Load pre-trained model (CNN using MFCC)
model_file = "mfcc_model.h5"
classifier = tf.keras.models.load_model(model_file)

# Folder containing the classes used in training
classes_folder = r"C:\Users\Administrador\Documents\meu_projeto_ecoauralia\train_audio"
labels = sorted(os.listdir(classes_folder))


def load_audio(path, rate=32000):
    return librosa.load(path, sr=rate)


def adjust_mfcc(mfcc, fixed_size=216):
    if mfcc.shape[1] < fixed_size:
        return np.pad(mfcc, ((0, 0), (0, fixed_size - mfcc.shape[1])), mode='constant')
    return mfcc[:, :fixed_size]


def extract_mfcc(audio_path):
    try:
        y, sr = load_audio(audio_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc = adjust_mfcc(mfcc)

        mfcc = np.expand_dims(mfcc, axis=-1)  # (13, 216, 1)
        mfcc = np.expand_dims(mfcc, axis=0)   # (1, 13, 216, 1)
        return mfcc
    except Exception as e:
        print(f"Oops! Couldn't process this audio. The reason was: {e}")
        return None


def classify_audio_mfcc(audio_path):
    mfcc = extract_mfcc(audio_path)
    if mfcc is None:
        return

    prediction = classifier.predict(mfcc)
    top_indices = np.argsort(prediction[0])[::-1][:3]  # top-3 predictions

    print("\n🔊 Model predictions:")
    for i, idx in enumerate(top_indices):
        species = labels[idx]
        confidence = prediction[0][idx]
        marker = "✅" if i == 0 else "→"
        print(f"{marker} {i+1}. {species} ({confidence:.1%})")

    if prediction[0][top_indices[0]] < 0.6:
        print("⚠️ Prediction with low confidence. There may be doubt between species.")

    return labels[top_indices[0]]

if __name__ == "__main__":
    path = input("Drag the (.ogg) file you want to identify here: ").strip('"')

    if not os.path.isfile(path):
        print("Hmm... it seems this file doesn't exist. Try again!")
    else:
        classify_audio_mfcc(path)
