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

To prepare the data for model training, the following steps were taken:

1. **Imputation and Cleaning**: Missing values in the dataset were filled forward and backward, then remaining NaNs were dropped. Columns with any remaining missing values were identified, and rows containing NaN were printed for inspection.

2. **Standardization**: Numeric columns were standardized using `StandardScaler`. The features to be scaled were selected dynamically based on the column index.

3. **Sequence Generation**: Time series data were transformed into input sequences with a fixed length of `seq_length` (100 in this case). Each sequence consisted of features from multiple timesteps, while the target was set as the 'Open' price at the end of the sequence.

4. **Train-Test Split**: 
   - First, data was split into a full training set (90% of the data) and a test set (10%) with a `train_test_split`, ensuring no shuffling to maintain temporal continuity.
   - The training set was further split into training and validation sets, with an 80-20 split, again maintaining temporal continuity.

5. **Sanity Checks**: 
   - NaN and infinite values were checked after all operations to ensure data quality.
   - Total lengths of train and test sets were verified to ensure data integrity.

6. **Save Cleaned Data**: The cleaned data was saved to `cleaned_stock_data.csv` for future reference and use.

**Summary of Dataset Shapes:**
- `X_train_full`: {X_train_full.shape}
- `X_test`: {X_test.shape}
- `X_train`: {X_train.shape}
- `X_val`: {X_val.shape}

Dates of the validation and test sets were printed to ensure the correct temporal split.

The data for the model was sourced from `APPL_after_feature_engineering_and_cleaning.csv`, which includes data from February 1, 2002, to April 26, 2024, and all feature engineering performed.

## Model Configuration

The model is built using PyTorch with the following architecture:

- **LSTM Layer**: 
  - The model has a 3-layer Bi-directional LSTM network (`nn.LSTM`) that takes 16 input features (`n_features`) and outputs 128 hidden features (`n_hidden`). 
  - The LSTM is set to `batch_first=True` to maintain batch dimension consistency, has a dropout of 0.2, and is bidirectional to capture sequential dependencies from both forward and backward directions.

- **Attention Mechanism**:
  - A custom attention mechanism (`Attention` class) is designed to focus on relevant parts of the input sequence, capturing the most important information for prediction. 
  - The attention mechanism computes scores (`attn_scores`) using matrix multiplication and applies softmax to obtain attention weights. The weighted sum of these scores is used to compute the attention output (`attn_output`).

- **Linear Layers**:
  - **First Linear Layer (`linear1`)**: Takes the attention output (256 features) and reduces it to 64 features.
  - **Second Linear Layer (`linear2`)**: Further reduces the 64 features to 32.
  - **Third Linear Layer (`linear3`)**: Outputs a single predicted value.

- **Residual Layer**: 
  - A residual layer (`residual_layer`) is used to add skip connections from the original input features (16) to the attention outputs. This connection adds the original input to the attention output, enabling the model to retain the initial information and improve training stability.

- **Weight Initialization**:
  - Custom weight initialization is applied to the linear layers and LSTM using the `init_weights` function, which applies the following:
    - Linear layers: Kaiming normal initialization and bias initialization to 0.01.
    - LSTM layers: Xavier initialization for input-hidden weights, orthogonal initialization for hidden-hidden weights, and bias initialization to zero.

The final model is instantiated using these configurations:
```python
model = StockPredictor(
    n_features=16,
    n_hidden=128,
    seq_len=100,  # The desired sequence length
    n_layers=3    # The number of LSTM layers
)

model.apply(init_weights)  # Apply weight initialization```

## Training Setup

The training setup for the model was as follows:

- **Optimizer and Loss**: 
  - **Optimizer**: The Adam optimizer was used with a learning rate of 0.00001 and a weight decay of 0.01 to help prevent overfitting.
  - **Combined Loss Function**: A custom loss function was designed to combine multiple loss types, helping the model learn more effectively by capturing different error metrics:
    - **Mean Squared Error (MSE) Loss**: Applied a weight of 0.05 to prioritize reducing large prediction errors.
    - **Smooth L1 Loss**: Weighted at 0.1 to reduce the influence of outliers by combining L1 and L2 losses.
    - **Mean Absolute Error (MAE) Loss**: The primary contributor with a weight of 0.45, emphasizing prediction accuracy.
    - **Quantile Loss**: Applied at a weight of 0.4 to improve prediction accuracy by considering the quantile of prediction errors.
- **Scheduler**:
  - **Learning Rate Scheduler**: The scheduler used was `torch.optim.lr_scheduler.ReduceLROnPlateau`.
  - **Configuration**: The scheduler was configured with a reduction factor of 0.5 and patience of 5 epochs. This means that if the validation loss did not improve after 5 epochs, the learning rate would be reduced by half.
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
