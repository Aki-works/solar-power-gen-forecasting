# ☀️ Deep Learning for Solar Power Generation Forecasting

Forecasting solar power output is a crucial task in the integration of renewable energy into modern power grids. This project investigates and compares advanced deep learning techniques, including hybrid and ensemble architectures to model and predict solar power generation using real-world multivariate time-series data.

---

## 📌 Project Overview

With the growing reliance on solar energy, accurate forecasting of solar power output is essential to maintain grid stability, reduce energy storage demands, and improve scheduling. However, solar power generation is highly variable due to environmental factors such as cloud cover, temperature, and irradiance.

This project explores several deep learning architectures that model both short-term and long-range temporal dependencies in solar power data, including:

- Long Short-Term Memory (LSTM) networks
- Convolutional Neural Networks (CNNs)
- Temporal Convolutional Networks (TCNs)
- Transformer models with self-attention
- CNN-LSTM hybrids (including time-distributed variants)
- Ensemble learning combining multiple models

---

## 📊 Dataset

- **Source**: [INESC TEC Smart Grid and Electric Vehicle Lab (SGEVL)](https://rdm.inesctec.pt/dataset/pe-2020-002)
- **Region**: Northern Portugal
- **Timeframe**: April 28, 2013 – June 28, 2016
- **Resolution**: Hourly readings
- **Features Used**:
  - Temperature (`temp`)
  - Cloud Cover (`cloud_cover`)
  - Shortwave Radiation (`shortwave_flux`)
  - Solar Power Output (`power`) - the target variable

Data cleaning included filtering null values, retaining zero-output records to capture diurnal patterns, and performing MinMax scaling.

---

## ⚙️ Methodology

1. **Feature Engineering**:
   - Lag features (1hr, 2hr, 3hr, 6hr)
   - Rolling statistics (mean, std)
   - Cyclical encodings (day, hour, month)

2. **Model Architectures**:
   - `LSTM`: Good for long-term dependencies
   - `CNN`: Captures local temporal patterns
   - `CNN-LSTM Hybrid`: Combines spatial & temporal features
   - `TimeDistributed CNN-LSTM`: Adds finer temporal segmentation
   - `TCN`: Uses dilated causal convolutions
   - `Transformer`: Captures global context via attention

3. **Ensemble Learning**:
   - Combined TCN and Transformer predictions using regression-based weighting
   - Outperformed all individual models

4. **Optimization**:
   - Optimizer: `AdamW`
   - Regularization: Dropout, EarlyStopping
   - Scheduler: `ReduceLROnPlateau`
   - Loss Function: MSE

---

## 🧪 Model Evaluation

Each model was evaluated on:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score** (Coefficient of Determination)

| Model                         | RMSE (W) | MAE (W) | R² Score |
|------------------------------|----------|---------|----------|
| LSTM                         | 990.73   | 507.35  | 0.928    |
| CNN                          | 1143.29  | 662.31  | 0.905    |
| Time-Distributed Hybrid      | 886.32   | 433.81  | 0.942    |
| Temporal Convolutional Net   | 869.86   | 405.92  | 0.9449   |
| Transformer                  | 873.85   | 424.49  | 0.9444   |
| **Ensemble (TCN + Transformer)** | **850.97**   | **390.54**  | **0.9488**  |

---

## 📌 Key Findings

- **CNN-LSTM hybrids** outperform individual CNN or LSTM models.
- **Transformer-based models** excel at long-sequence forecasting and capturing irregular patterns.
- **TCNs** offer faster training and stable performance for large input sequences.
- **Ensemble models** significantly improve accuracy and generalization by leveraging strengths of individual models.

---

## 🚀 Installation & Usage

### Prerequisites
- Python ≥ 3.8
- TensorFlow ≥ 2.x or PyTorch (for Transformer/TCN)
- Jupyter Notebook / Colab
- scikit-learn, pandas, matplotlib, seaborn
