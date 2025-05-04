"""
train_mfcc_model.py

I chose to use MFCCs (Mel-Frequency Cepstral Coefficients) to train a model for classifying animal sounds.
This method is widely used in audio analysis, especially in speech recognition and environmental sound classification, such as animal sounds.
MFCCs represent audio frequency in a way similar to human auditory perception, capturing important aspects and ignoring irrelevant noise.
They are extracted from audio files, such as .ogg, and transformed into matrices that represent sound characteristics.

After that, the model training is done, as per the following code:
"""

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Directory with audios separated by species
diretorio_audios = r"C:\Users\Administrador\Documents\meu_projeto_ecoauralia\train_audio"
species = sorted(os.listdir(diretorio_audios))

mfcc_list = []
labels = []

print("🔍 Extracting MFCCs from audio files...")

for idx, specie in enumerate(species):
    specie_path = os.path.join(diretorio_audios, specie)

    if not os.path.isdir(specie_path):
        continue

    for audio in os.listdir(specie_path):
        if audio.endswith(".ogg"):
            audio_path = os.path.join(specie_path, audio)

            try:
                y_audio, sr = librosa.load(audio_path, sr=32000)
                mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
                mfcc = mfcc[:, :216]  # Ensure same size

                if mfcc.shape[1] < 216:
                    mfcc = np.pad(mfcc, ((0, 0), (0, 216 - mfcc.shape[1])), mode='constant')

                mfcc_list.append(mfcc)
                labels.append(idx)

            except Exception as e:
                print(f"Error processing '{audio}': {e}")

X = np.array(mfcc_list)
y = np.array(labels)
X = np.expand_dims(X, axis=-1)

print(f"✅ Extraction completed: Data {X.shape}, Labels {y.shape}")

# Split data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Adjust weights for imbalanced classes
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(zip(np.unique(y_train), weights))

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(13, 216, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(species), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(
    X_train, y_train,
    epochs=40,
    validation_data=(X_valid, y_valid),
    class_weight=class_weights
)

# Save trained model
model.save("mfcc_model.h5")
print("✅ Model saved as 'mfcc_model.h5'!")
