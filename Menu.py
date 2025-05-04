import os
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from classifier import classify_spectrogram, generate_description, search_real_image
from mfcc_classifier import classify_audio_mfcc

# Automatically generate spectrogram
def generate_spectrogram(audio_path):
    audio_name = Path(audio_path).stem
    output_folder = Path("spectrograms/test")
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{audio_name}.png"

    try:
        y, sr = librosa.load(audio_path, sr=32000)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_DB, sr=sr, x_axis=None, y_axis=None)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return str(output_path)
    except Exception as e:
        print(f"Oops, couldn't create the spectrogram: {e}")
        return None

# Main menu

def menu():
    while True:
        print("\n🌿 ECOAURALIA 🌿")
        print("1 - Classify audio by Spectrogram")
        print("2 - Classify audio by MFCC")
        print("3 - Get species description")
        print("4 - Search real image of the species")
        print("5 - Execute complete flow (audio -> spectrogram -> text -> image)")
        print("6 - Exit")

        choice = input("👉 Choose an option: ")

        if choice in ["1", "2"]:
            test_folder = Path(r"C:\Users\Administrador\Documents\meu_projeto_ecoauralia\train_audio\TesteAudios\convertidos")
            audios = list(test_folder.glob("*.ogg"))

            if not audios:
                print("⚠️  No audio found in the test folder.")
                continue

            print("\nAvailable audios:")
            for idx, audio in enumerate(audios):
                print(f"{idx + 1} - {audio.name}")

            audio_idx = input("Audio number: ")
            if audio_idx.isdigit() and 1 <= int(audio_idx) <= len(audios):
                audio_path = str(audios[int(audio_idx) - 1])
                if choice == "1":
                    img_path = generate_spectrogram(audio_path)
                    if img_path:
                        classify_spectrogram(img_path)
                else:
                    classify_audio_mfcc(audio_path)
            else:
                print("⚠️ Invalid option!")

        elif choice == "3":
            species = input("Enter the desired species: ")
            generate_description(species)

        elif choice == "4":
            species = input("Enter the desired species: ")
            search_real_image(species)

        elif choice == "5":
            test_folder = Path(r"C:\Users\Administrador\Documents\meu_projeto_ecoauralia\train_audio\TesteAudios\convertidos")
            audios = list(test_folder.glob("*.ogg"))

            if not audios:
                print("⚠️  No audio found in the test folder.")
                continue

            print("\nAvailable audios:")
            for idx, audio in enumerate(audios):
                print(f"{idx + 1} - {audio.name}")

            audio_idx = input("Audio number for complete flow: ")
            if audio_idx.isdigit() and 1 <= int(audio_idx) <= len(audios):
                audio_path = str(audios[int(audio_idx) - 1])
                img_path = generate_spectrogram(audio_path)
                if img_path:
                    species = classify_spectrogram(img_path)
                    if species:
                        generate_description(species)
                        search_real_image(species)
            else:
                print("⚠️ Invalid option!")

        elif choice == "6":
            print("🌱 Thank you for using EcoAuralia! See you later!")
            break

        else:
            print("⚠️ Invalid option! Try again.")

if __name__ == "__main__":
    menu()
