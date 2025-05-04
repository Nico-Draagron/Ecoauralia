# 🌿 EcoAuralia: Acoustic Animal Recognition with AI

## 📘 Introduction  
**EcoAuralia** is a lightweight command-line AI tool, developed as the first graded assignment for the IA 1 course. It “listens” to short `.ogg` audio clips of farm animals and:

- **Classifies** them as **dog**, **cat**, **cow**, **pig** or **sheep**  
- **Generates** a detailed natural-language description via Google’s Gemini API  
- **Launches** a real-world image search in your browser  

Under the hood it uses two complementary approaches:
1. **Spectrogram + MobileNetV2**: converts 5 s audio into 128×128 Mel-spectrograms and fine-tunes a pretrained MobileNetV2 head.  
2. **MFCC + Custom CNN**: extracts 13 MFCCs (padded/cropped to 13×216) and feeds them to a small 2-layer convolutional network.

---

## 🧰 Dependencies

| Library               | Purpose                                                |
|-----------------------|--------------------------------------------------------|
| TensorFlow / Keras    | Model definition & training (CNNs)                     |
| Librosa               | Audio I/O & feature extraction (spectrograms, MFCCs)   |
| NumPy                 | Numerical array operations                             |
| Matplotlib            | Rendering & saving spectrogram images                  |
| google-generativeai   | Text description generation (Gemini)                   |
| webbrowser            | Opening Google image search in the default browser     |
| (Optional) diffusers  | Local image synthesis via Hugging Face (disabled now)  |

Install with:
\`\`\`bash
pip install tensorflow librosa matplotlib numpy google-generativeai
\`\`\`

---

## 🔧 Project Structure

\`\`\`
ecoauralia/
├── classificador.py               # Spectrogram classification + Gemini + image search
├── classificador_mfcc.py          # MFCC-based audio classification
├── menu_principal.py              # CLI menu interface
├── model_mobilenet_finetuned.h5   # Spectrogram model (MobileNetV2 head)
├── model_mfcc.h5                  # MFCC CNN model
├── train_audio/                   # Training audio files organized by species
└── spectrograms/                  # Generated spectrogram images for train/test
\`\`\`

---

## ⚙️ Core Features

### 1. Spectrogram Classification  
- Converts a 5 s \`.ogg\` clip into a 128×128 Mel-spectrogram image  
- Classifies with a fine-tuned MobileNetV2 head  

### 2. MFCC Classification  
- Extracts 13 MFCC coefficients, fixed to a 13×216 shape  
- Classifies with a small 2-layer Conv2D network  

### 3. Text Description (Gemini)  
- Sends the predicted species name to Google Gemini  
- Prints a concise, informative description  

### 4. Real-World Image Search  
- Opens your browser to a Google Images query for the predicted species  

---

## ▶️ Getting Started

1. **Clone the repo** and \`cd ecoauralia/\`  
2. **Install dependencies**  
   \`\`\`bash
   pip install tensorflow librosa matplotlib numpy google-generativeai
   \`\`\`  
3. **Run the CLI menu**  
   \`\`\`bash
   python menu_principal.py
   \`\`\`  
4. **Follow the prompts** to choose classification, description, image search or the full pipeline.

---

## 🧪 Example Session

\`\`\`
🌿 ECOAURALIA 🌿
1 – Classify by Spectrogram (.ogg)
2 – Classify by MFCC       (.ogg)
3 – Get Text Description
4 – Search Real Images
5 – Full Pipeline
6 – Exit
👉 Choose an option: 1
\`\`\`

---

## 💡 Notes & Considerations

- Training data is organized by species folders under \`train_audio/\`.  
- Spectrogram model fine-tuned with data augmentation and class-weighting to handle imbalance.  
- MFCC model trained for 40 epochs with sparse categorical cross-entropy and balanced classes.  
- Image synthesis via Diffusers was removed to keep the project lightweight and fully local.  
- Audio quality and class balance directly impact classification accuracy.  
- Gathering clean, diverse animal audio samples can be challenging.

---

> Developed as the first evaluative assignment in the IA 1 course.
