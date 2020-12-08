# Emotion recognition from audio
Just a small playground I scraped together for AER

## Models/Ideas
- Old school classification (i.e. Decision Tree)
- Manually built 1D CNN on spectrogram data
- Transfer Learning from VGG16 (todo: VGG19) on spectrogram images

## Achieved Results
- Decision Trees at around 30% accuracy
- 1D CNN at around 50% accuracy
- Transfer Learning methods tbd.

All without outrageous outliers in Precision/Recall/F1

## Dataset
Kaggle dataset: https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio
Set in folder `assets/archive`

Data for the transfer learning approach is generated from those audio files and are located in `plots`