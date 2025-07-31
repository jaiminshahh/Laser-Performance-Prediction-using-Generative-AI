# Laser-Performance-Prediction-using-Generative-AI

This repository contains models and scripts used to predict laser performance characteristics using generative AI techniques. The architecture comprises multiple components, each handling different stages of the prediction pipeline.

---

## 📦 Components & How to Run

### 1. ViT-3D Convolver
- **Input:** Original spatial beam and pulse data  
- **Output:** Intermediate spatial beam and pulse predictions  

**To run:**
1. Install dependencies: `PyTorch`, `pandas`, `sklearn`
2. Place the original data in the specified directories
3. Run:  
   ```bash
   python main.py

### 2. Spatial Beam Denoiser
- **Input:** Intermediate spatial data from ViT-3D Convolver
- **Output:** Refined spatial beam predictions
- 
**To run:**
1. Ensure dependencies (PyTorch, pandas, scikit-learn) are installed
2. Place intermediate data in the appropriate directory
3. Run the script:
   ```bash
   python unet_main.py

### 3. Pulse Denoiser
- **Input:** Intermediate spatio-temporal data
- **Output:** Refined pulse predictions

**To run:**
1. Install required dependencies
2. Place intermediate data in the correct location
3. Execute the denoiser training script:
   ```bash
   python pulse_denoiser.py

### 4. SFFN-AE (Spatio-Temporal Feature Fusion Network - Autoencoder)
- **Input:** Original pulse shape data (energy-normalized and truncated to the first 59 points)
- **Output:** Predicted pulse shapes compared to ground truth

**To run:**
1. Install dependencies (PyTorch, pandas, scikit-learn)
2. Place pulse data in the specified directory
3. Run the SFFN-AE script:
   ```bash
   python sffn_ae.py

### 5. LSTM Model
- **Input:** Original pulse shape data (energy-normalized and truncated to the first 59 points)
- **Output:** Predicted pulse shapes compared to ground truth

**To run:**
1. Install all required dependencies
2. Place pulse data in the specified directory
3. Run the LSTM model script:
 ```bash
 python lstm_model.py
