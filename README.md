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
Kaggle dataset: https://www.notion.so/Datasets-516aa813208e4786a86c59a05780fde7#0f32419dc10043d5bea93052b6a816f2
Set in folder `assets/archive`

Data for the transfer learning approach is generated from those audio files and are located in `plots`