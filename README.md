# Apple Stock Prediction Model

## Overview

This project aims to train a machine learning model to predict Apple's stock opening price using historical data from February 1, 2002, to April 26, 2024.

## Feature Engineering

The feature engineering process involved:
1. **Filtering and Cleaning Data**: Rows containing the string 'Dividend' in the 'Open' column were filtered out. Numeric columns were converted to numeric types.
2. **Moving Averages**: Calculated Simple Moving Average (SMA) and Exponential Moving Averages (EMA) over specified windows.
3. **RSI Calculation**: Computed the 14-day Relative Strength Index (RSI).
4. **MACD Calculation**: Calculated the Moving Average Convergence Divergence (MACD) and its signal line.
5. **Bollinger Bands**: Created Bollinger Bands using SMA and standard deviations.
6. **VWAP Calculation**: Calculated the Volume Weighted Average Price (VWAP).
7. **Williams %R**: Although Williams %R was calculated, its use led to a decline in model performance, producing predictions that were contrary to actual trends. Therefore, it was ultimately not used. Its result saved in `william.csv` 

## Data Preprocessing

To prepare the data for model training:
1. **Imputation and Cleaning**: Missing values were filled forward and backward, and remaining NaNs were dropped.
2. **Standardization**: Scaled the data using `StandardScaler`.
3. **Sequence Generation**: Generated sequences of a specified length from the time series data to feed into the model.
4. **Train-Test Split**: The data was split into training, validation, and test sets, ensuring temporal continuity.

## Model Configuration

The model is built using PyTorch with the following architecture:
- **LSTM Layer**: A 3-layer Bi-directional LSTM with a hidden size of 128 and a dropout of 0.2.
- **Attention Mechanism**: A custom attention layer to enhance important features.
- **Linear Layers**: Three linear layers with residual connections for feature extraction and regression.
- **Residual Layer**: Adds skip connections between input features and attention outputs.

## Training Setup

The training setup involved:
- **Optimizer and Loss**: Adam optimizer with a custom loss function combining MSE, MAE, quantile loss, and Smooth L1 loss.
- **Scheduler**: Learning rate scheduler to adjust the learning rate based on validation loss.
- **Hyperparameters**: Batch size of 4, 50 epochs, learning rate of 0.00001, and weight decay of 0.01.
- **Training Environment**: Trained using A100 GPUs on Google Colab.

## File Descriptions

1. **Best Model**: The best model based on testing MAPE is `best_model_20240502104451_epoch_27_val_loss_0.0101.pt`.
2. **Alternative Model**: The model with better validation prediction results is `model_with_best_validation_20240502092010_epoch_8_val_loss_0.0457.pt`. Key differences include:
   - **Attention Mechanism**: Utilizes multi-head attention instead of the basic attention mechanism.
   - **Normalization**: Incorporates layer normalization in some layers.
   - **Residual Connections**: More extensive use of residual connections in linear layers.
   - More details for this model's architecture see `best_validation_model_architecture.py`
3. **R Code Attempts**: 
   - **ARIMA and GARCH Models**: Tried fitting ARIMA and GARCH models to the data, but both models failed to converge or fit the data well. This led to the exploration of machine learning models.
   - **Random Forest Model**: Also attempted a Random Forest model but found its predictions to be less accurate than desired.
4. **Data Source**: The training data for the model comes from `APPL_after_feature_engineering_and_cleaning.csv`, which contains data from February 1, 2002, to April 26, 2024, and includes all feature engineering performed.
