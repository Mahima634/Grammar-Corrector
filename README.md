# Grammar Scoring Engine — SHL Hiring Assessment 2026

## Problem
Predict MOS Likert Grammar Score (0-5) from 45-60 second spoken audio samples.

## Approach
Multimodal feature extraction combining linguistic and acoustic signals:

- **Text Features (8):** Whisper ASR transcripts + LanguageTool grammar errors, error density, type-token ratio, filler words, sentence variance, avg word length, repetition
- **Audio Features (768):** WavLM-base-plus embeddings
- **Acoustic Features (40):** MFCC mean + std (20 coefficients)
- **Total:** 816 features per audio sample

## Pipeline
1. **Audio Preprocessing:** `librosa.load(sr=16000)`
2. **ASR:** Whisper (small/medium) via GPU
3. **Grammar Check:** LanguageTool (en-US)
4. **Audio Embeddings:** `microsoft/wavlm-base-plus` (768-dim)
5. **Model:** Ensemble of LightGBM, XGBoost, and CatBoost regressors
6. **Evaluation:** RMSE + Pearson Correlation

## Results
- **Training RMSE:** 0.0752 (Ensemble)
- **Validation RMSE:** 0.5977
- **Validation Pearson:** 0.8994
- **Kaggle Public Score:** **0.5034** (Best)
- **Rank:** ~278 (out of ~336 participants)

## Key Insights
- WavLM embeddings contribute the most to model performance
- MFCC features provide complementary acoustic signals
- Text features (grammar errors) have minimal but useful impact
- **Ensemble (LGB + XGB + CatBoost)** significantly improves generalization compared to single LightGBM (0.5353 → 0.5034)
- Feature selection (Top 100) was tested but did not improve the score

## Files
- `Grammar corrector.ipynb` — Full notebook with feature extraction, model training, and evaluation
- `submission_ensemble.csv` — Final submission file (Score: 0.5034)

## Author
Mahima Prakash ([@Mahima634](https://github.com/Mahima634))
