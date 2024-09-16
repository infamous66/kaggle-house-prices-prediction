# House Price Prediction - README

This repository contains the code and methodology used to predict house prices using the dataset from Kaggle's "House Prices - Advanced Regression Techniques" competition. The project follows various machine learning practices, including data preprocessing, feature engineering, model selection, and evaluation. Below is an overview of the steps and challenges encountered throughout the project.

## Project Structure

- **Preprocessing:**
  - **Handling Missing Data:** Both numerical and categorical columns had missing values. For numerical columns, missing values were replaced with mean values, while categorical columns were filled with mode values, to ensure no information was lost.
  - **Outlier Detection and Handling:** For features with high skewness (e.g., `GrLivArea`, `TotalBsmtSF`), we applied log transformations to reduce their impact.
  - **Dropping Weak Features:** Features like `BedroomAbvGr`, `MoSold`, and `YrSold` had weak correlations with the target variable `SalePrice`, and were dropped to reduce noise.
  - **Scaling:** Numerical features were standardized using a standard scaler to improve the performance of the model.

- **Feature Engineering:**
  - **One-Hot Encoding:** Categorical columns like `Neighborhood`, `Condition1`, and `SaleType` were one-hot encoded to create binary features for better model interpretation.
  
- **Model Selection:**
  - **Ridge Regression:** A Ridge regression model was selected to handle multicollinearity, which was a significant concern due to the highly correlated nature of certain features.


- **Challenges and Solutions:**
  - **Multicollinearity:** Due to high correlations between several features, Ridge regression was chosen to penalize the weights of correlated features.
  - **Imbalanced Data:** Some features had very few unique values (e.g., `PoolQC`, `Fence`). These were either dropped or grouped to reduce their impact on the model.
  - **High Skewness:** Log transformations were applied to features with high skewness to normalize their distributions and improve the model’s performance.
  - **Feature Scaling:** Standard scaling was necessary to ensure the model could handle the varying ranges of different features.
  
- **Model Evaluation:**
  - The model was evaluated using the Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score. The best performance was achieved using the Ridge regression model with an MSE of **504,314,193.88323164**, and an R² score of **0.89**.

## How to Run

1. Clone the repository.
2. Install the required dependencies listed in the `requirements.txt`.
3. Run the Jupyter notebook file `house_price_pred.ipynb` to see the results.

## Conclusion

This project highlighted several key steps in the machine learning workflow, including dealing with missing data, handling skewed features, managing multicollinearity, and carefully selecting the right model for the problem. The final Ridge regression model performed well in predicting house prices, achieving strong validation results.