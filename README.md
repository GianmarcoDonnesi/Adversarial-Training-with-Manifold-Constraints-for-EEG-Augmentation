# Adversarial-Training-with-Manifold-Constraints-for-EEG-Augmentation

Welcome to the repository containing the code developed for the **Elective in AI (EAI)** course (A.Y. 24/25). This project focuses on generating **synthetic EEG data** using a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** to enhance the performance of a **Temporal Fusion Transformer (TFT)** for classification and regression tasks on EEG signals. This repository contains scripts for data preprocessing, WGAN-GP implementation, with its training and validation part, TFT implementation, and validation, offering a complete pipeline for advanced EEG data augmentation and analysis.

---

## **Table of Contents**
1. [Project Structure](#project-structure)
2. [System Requirements and Dependencies](#system-requirements-and-dependencies)
3. [Dataset and Preprocessing](#dataset-and-preprocessing)
4. [WGAN-GP](#wgan-gp)
5. [Temporal Fusion Transformer (TFT)](#temporal-fusion-transformer-tft)
6. [Execution Instructions](#execution-instructions)
7. [Validation Metrics](#validation-metrics)
8. [Authors and License](#authors-and-license)

---

## **Project Structure**

The project files are organized as follows:

```
Adversarial-Training-with-Manifold-Constraints-for-EEG-Augmentation-main/
│
├── load_preprocessed_data.py       # Loads preprocessed .npz data
├── prepare_tft_data.py             # Prepares data for TFT (min-max scaling, DataLoader creation)
├── prepare_wgan_data.py            # Prepares data for WGAN-GP (z-score normalization, DataLoader creation)
├── preprocessing_data.py           # EEG preprocessing pipeline (filtering, ICA, epoching, SPD matrices)
├── riemannian_manifold.py          # Utility functions for Riemannian operations (SPD matrices, tangent projection)
├── tft_model.py                    # Implementation of the TFT model
├── train_models.py                 # Complete training pipeline (WGAN-GP and TFT)
├── validate_models.py              # Validation script for WGAN and TFT
├── wgan_gp.py                      # Implementation of the WGAN-GP
│
└── README.md                       
```

---

## **System Requirements and Dependencies**

- **Python 3.8+**  
- **PyTorch 1.10+**  
- **Torchvision** (optional for related utilities)  
- **CUDA** (optional but strongly recommended for faster training)  
- **NumPy**  
- **Scikit-learn**  
- **Matplotlib** (for optional visualizations)  
- **MNE** (for EEG preprocessing)  
- **PyRiemann** (for SPD tangent projection)  
- **pandas** (optional for tabular data management)  
- **AutoReject** (for automatic epoch cleaning)  
- **tqdm** (for progress bars)  

> **Tip**: Using a **virtual environment** (e.g., via `conda` or `venv`) is recommended to isolate dependencies.

---

## **Dataset and Preprocessing**

### **Dataset**
- **Name**: *Human infant EEG recordings for 200 object images presented in rapid visual streams*  
- **Source**: [NEMAR](https://nemar.org/dataexplorer/detail?dataset_id=ds005106)  
- **Characteristics**:
  - 42 infants
  - 32 EEG channels
  - 200 visual stimuli
- **Format**: BIDS-EEG  
- **Path (Example)**:  
  ```
  ./ds005106
  ```

### **Preprocessing**
1. **Filtering**: band-pass filtering at 0.5–40 Hz (IIR filter).  
2. **ICA and Referencing**: artifact removal and common average referencing.  
3. **Interpolation**: detect and interpolate noisy channels.  
4. **Segmentation (Epoching)**: epochs of 0.0–0.5 seconds.  
5. **Cleaning**: using **AutoReject** (with “majority” consensus).  
6. **Feature Extraction**:
   - Compute **SPD covariance matrices**.
   - Perform **tangent space projection** using **PyRiemann**.  
7. **Saving**: preprocessed data is saved in `.npz` format in the directory:
   ```
   ./ds005106/derivatives/preprocessing
   ```
   Main script: **`preprocessing_data.py`**

---

## **WGAN-GP**

### **Objective**
Generate realistic synthetic EEG data to enhance TFT training.

### **Architecture**
- **Generator (G)**: takes a noise vector (`nz`) as input and generates synthetic EEG sequences of size `[batch_size, 1, 528]`.
- **Discriminator (D)**: evaluates the authenticity of EEG sequences (real or synthetic) using 1D Convolutions and Batch Normalization.
- **Training with Gradient Penalty (GP)**: during the training phase, a gradient constraint is imposed to stabilize the training process.

### **Key Scripts**
- **`prepare_wgan_data.py`**: 
  - Normalizes features using **z-score**.
  - Creates a `DataLoader` for WGAN training.
- **`wgan_gp.py`**:
  - Implement **WGAN-GP** in Pytorch, defining the **Generator**, the **Discriminator**, the **Gradient Penalty Trainer** and their functions

---

## **Temporal Fusion Transformer (TFT)**

### **Objective**
Train a **Transformer-based** model on EEG sequences (real or synthetic) for **classification** or **regression** tasks.

### **Architecture**
- **Encoder**:  
  - Multi-layer LSTM.  
  - **Multi-Head Self-Attention**.  
  - **Positional Encoding**.  
  - **Gated Residual Networks (GRN)**.
- **Decoder**:  
  - Single-layer LSTM.  
  - GRNs for output.  
- **Tasks**: 
  - Classification (e.g., recognizing a stimulus).  
  - Regression (e.g., predicting future EEG activity).

### **Key Scripts**
- **`prepare_tft_data.py`**:  
  - Performs **min-max scaling**.  
  - Creates temporal windows (sequencing).  
  - Generates a `DataLoader` for training/validation.
- **`tft_model.py`**:  
  - Implements the **Temporal Fusion Transformer** in PyTorch.

---

## **Execution Instructions**

### **1. Clone or Download the Repository**
```bash
git clone https://github.com/GianmarcoDonnesi/Adversarial-Training-with-Manifold-Constraints-for-EEG-Augmentation
cd Adversarial-Training-with-Manifold-Constraints-for-EEG-Augmentation-main
```

### **2. Install Dependencies**
It is recommended to use a virtual environment:
```bash
conda create -n Adversarial-Training-with-Manifold-Constraints-for-EEG-Augmentation-main python=3.8
conda activate Adversarial-Training-with-Manifold-Constraints-for-EEG-Augmentation-main
```

All required dependencies are listed in the `requirements.txt` file and can be installed using:
```bash
pip install -r requirements.txt
```
This will ensure that all the necessary libraries (e.g., PyTorch, MNE, PyRiemann, AutoReject) are installed in the correct versions for the project.

### **3. Preprocess EEG Data (Optional if Already Preprocessed)**
If you don't have preprocessed `.npz` data, run:
```bash
python preprocessing_data.py
```
- Ensure the BIDS-EEG dataset path is configured in `preprocessing_data.py`.

### **4. Load Preprocessed Data**
To load and combine the preprocessed data into a usable format:
```bash
python load_preprocessed_data.py
```
- This script will generate a unified dataset for subsequent training.

### **5. Prepare Data for WGAN**
```bash
python prepare_wgan_data.py
```
- Performs **z-score normalization** and creates a `DataLoader` for WGAN.

### **6. Train WGAN-GP**
```bash
python train_models.py --train_wgan
```
- This script executes the complete WGAN-GP training pipeline using parameters defined in `wgan_gp.py`.

### **7. Prepare Data for TFT**
```bash
python prepare_tft_data.py
```
- Performs **min-max scaling** and sequences the data for TFT.

### **8. Train TFT**
```bash
python train_models.py --train_tft
```
- Trains the TFT model using real and/or synthetic data, depending on script configurations.

### **9. Validation**
```bash
python validate_models.py
```
- Calculates metrics (FID, IS, Accuracy, F1-score, etc.) to evaluate both WGAN and TFT.

> **Note**: Use flags (or internal script modifications) to define whether to train/validate **only** WGAN, **only** TFT, or **both**.

---

## **Validation Metrics**

1. **For WGAN-GP**:
   - **FID** (Fréchet Inception Distance)
   - **IS** (Inception Score)
   - **KID** (Kernel Inception Distance)
   - **Precision & Recall** (for sample diversity)

2. **For TFT**:
   - **Accuracy**  
   - **F1-score**  
   - **MSE / RMSE** (for regression tasks)  
   - **Confusion Matrix** (for classification tasks)

---

## **Authors and License**

- **Authors**:  
  - Gianmarco Donnesi | [donnesi.2152311@studenti.uniroma1.it](mailto:donnesi.2152311@studenti.uniroma1.it)  
  - Michael Corelli | [corelli.1938627@studenti.uniroma1.it](mailto:corelli.1938627@studenti.uniroma1.it)

- **License**: this project is licensed under the [GPL-3.0 License](LICENSE). See the file for more details.

For questions, suggestions, or bug reports, open an *Issue* or submit a *Pull Request* on GitHub.
