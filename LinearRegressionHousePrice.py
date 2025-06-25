import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

def load_data(file_path):
    """
    Loads house price data from a CSV file into a Pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure 'house_prices.csv' is in the same directory as the script.")
        # Create a dummy CSV for demonstration if not found
        print("Creating a dummy 'house_prices.csv' for demonstration purposes.")
        dummy_data = {
            'sqft_living': [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3200],
            'price': [200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000, 650000]
        }
        pd.DataFrame(dummy_data).to_csv(file_path, index=False)
        print("Dummy 'house_prices.csv' created. Please replace it with your actual data.")

    return pd.read_csv(file_path)

def preprocess_data(dataframe, feature_col, target_col):
    """
    Preprocesses the data for linear regression.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame.
        feature_col (str): The name of the feature column (e.g., 'sqft_living').
        target_col (str): The name of the target column (e.g., 'price').

    Returns:
        tuple: A tuple containing (features (X), target (y)) as NumPy arrays.
    """
    # Extract series and reshape them for scikit-learn
    # .reshape(-1, 1) converts a 1D array into a 2D array with one column.
    # The -1 means "calculate the size of this dimension automatically".
    x = np.array(dataframe[feature_col]).reshape(-1, 1)
    y = np.array(dataframe[target_col]).reshape(-1, 1)
    return x, y

def train_model(x_data, y_data):
    """
    Trains a Linear Regression model.

    Args:
        x_data (numpy.ndarray): The feature data.
        y_data (numpy.ndarray): The target data.

    Returns:
        sklearn.linear_model.LinearRegression: The trained model.
    """
    model = LinearRegression()
    model.fit(x_data, y_data)
    return model

def evaluate_model(model, x_data, y_data, y_pred):
    """
    Evaluates the linear regression model and prints performance metrics.

    Args:
        model (sklearn.linear_model.LinearRegression): The trained model.
        x_data (numpy.ndarray): The feature data used for training.
        y_data (numpy.ndarray): The actual target data.
        y_pred (numpy.ndarray): The predicted target data.
    """
    # Calculate Root Mean Squared Error (RMSE)
    regression_model_mse = mean_squared_error(y_data, y_pred)
    rmse = math.sqrt(regression_model_mse)
    print(f'Root Mean Squared Error (RMSE) = {rmse:.2f}')

    # Calculate R-squared value
    # model.score returns the coefficient of determination R^2 of the prediction.
    # It provides a measure of how well future samples are likely to be predicted by the model.
    r_squared = model.score(x_data, y_data)
    print(f'R-squared value = {r_squared:.4f}')

    # Get the coefficients (b1 and b0)
    b1 = model.coef_[0][0]
    b0 = model.intercept_[0]
    print(f'Coefficients: b1 (slope) = {b1:.2f}, b0 (intercept) = {b0:.2f}')
    print(f'The linear equation is: Price = {b1:.2f} * Size + {b0:.2f}')


def visualize_results(x_data, y_data, model, title, xlabel, ylabel):
    """
    Visualizes the data and the fitted linear regression model.

    Args:
        x_data (numpy.ndarray): The feature data.
        y_data (numpy.ndarray): The actual target data.
        model (sklearn.linear_model.LinearRegression): The trained model.
        title (str): The title for the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='green', label='Actual Data Points', alpha=0.6)
    plt.plot(x_data, model.predict(x_data), color='black', label='Regression Line')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    CSV_FILE_PATH = 'house_prices.csv'
    FEATURE_COLUMN = 'sqft_living'
    TARGET_COLUMN = 'price'

    # 1. Load Data
    house_data_df = load_data(CSV_FILE_PATH)
    if house_data_df is None:
        exit() # Exit if data loading failed (e.g., file not found and no dummy created)

    # 2. Preprocess Data
    X, y = preprocess_data(house_data_df, FEATURE_COLUMN, TARGET_COLUMN)

    # 3. Train Model
    linear_model = train_model(X, y)

    # 4. Make Predictions
    y_predictions = linear_model.predict(X)

    # 5. Evaluate Model
    print("\n--- Model Evaluation ---")
    evaluate_model(linear_model, X, y, y_predictions)

    # 6. Visualize Results
    print("\n--- Generating Plot ---")
    visualize_results(X, y, linear_model,
                      f"Linear Regression: House Price vs. {FEATURE_COLUMN.replace('_', ' ').title()}",
                      f"{FEATURE_COLUMN.replace('_', ' ').title()} (sqft)",
                      "Price ($)")
