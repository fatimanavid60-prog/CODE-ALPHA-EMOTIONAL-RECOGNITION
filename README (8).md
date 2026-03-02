# Emotion Recognition from Speech

## Overview
This project builds a deep learning model that recognizes human emotions from speech audio recordings. Audio files are processed to extract acoustic features, which are then fed into an LSTM neural network to classify emotions like happy, sad, angry, fearful, neutral, and more.

---

## Dataset Used
**RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song**
- Available on Kaggle: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
- 24 professional actors, 8 emotional categories
- Emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Audio format: .wav files at 48kHz

---

## Models Applied
**Stacked LSTM (Long Short-Term Memory)** neural network:
- 2 LSTM layers (256 and 128 units)
- Batch Normalization and Dropout for regularization
- Dense output layer with Softmax activation
- Optimizer: Adam | Loss: Categorical Cross-Entropy
- Callbacks: Early Stopping + Learning Rate Reduction

---

## Key Results and Findings

The model achieved solid classification performance across most emotion categories. Emotions with distinct acoustic patterns such as angry and happy were classified with the highest accuracy. Calm and neutral were the most commonly confused pair due to their acoustic similarity.

**Feature Engineering Summary:**

| Feature | Dimensions | What it captures |
|---------|-----------|-----------------|
| MFCCs | 40 | Timbral texture and tonal quality of speech |
| Chroma | 12 | Pitch class and harmonic content |
| Mel Spectrogram | 128 | Frequency content over time |
| **Total** | **180** | Combined acoustic fingerprint per sample |

**Key findings:**
- MFCC features were the most discriminative for emotion classification
- Angry and happy emotions showed the clearest acoustic separation
- Calm vs neutral was the hardest pair to distinguish
- Early stopping prevented overfitting effectively on the limited dataset size

---

## How to Run

1. Download the RAVDESS dataset from Kaggle and extract into a folder called `ravdess_data/`
2. Install dependencies:
```bash
pip install librosa tensorflow scikit-learn matplotlib seaborn pandas numpy
```
3. Run the notebook:
```bash
jupyter notebook "emotion recognition from speech.ipynb"
```

If the dataset folder is not found, the notebook will automatically use synthetic data so all cells can still be run end-to-end.
