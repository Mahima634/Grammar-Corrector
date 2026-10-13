# Grammar Scoring Engine — SHL Hiring Assessment 2026

## Problem
Predict MOS Likert Grammar Score (0-5) from 45-60 second audio samples.

## Approach
Multimodal feature extraction combining:
- **Text Features (8)**: Whisper ASR + LanguageTool grammar errors
- **Audio Features (768)**: WavLM-base-plus embeddings
- **Acoustic Features (40)**: MFCC mean + std

**Total: 816 features per audio sample**

## Pipeline
1. **Audio Preprocessing**: librosa.load(sr=16000)
2. **ASR**: Whisper (small/medium) via GPU
3. **Grammar Check**: LanguageTool (en-US)
4. **Audio Embeddings**: WavLM-base-plus (768-dim)
5. **Model**: LightGBM Regressor (1000 estimators, lr=0.02, depth=8)
6. **Evaluation**: RMSE + Pearson Correlation

## Results
- **Training RMSE**: (fill this)
- **Validation RMSE**: 0.5997
- **Validation Pearson**: 0.8983
- **Kaggle Public Score**: 0.5353

## Files
- `Grammar corrector.ipynb` — Full notebook
- `submission_v2.csv` — Final submission file

## Author
Mahima Prakash (@Mahima634)
