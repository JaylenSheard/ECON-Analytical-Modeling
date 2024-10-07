# Insurance Premium Analysis

This project analyzes insurance premium data to develop predictive models and provide recommendations on pricing strategies. The key factors in this analysis include smoker status, BMI, region, and other demographic variables. The notebook uses various regression techniques to determine how these factors influence insurance premiums.

## Project Overview

In this project, we explore:
- Data cleaning and preprocessing, including handling categorical variables.
- Exploratory Data Analysis (EDA) to visualize key trends.
- Regression analysis to predict insurance premiums based on smoker status, BMI, and other variables.
- Comparison between linear and polynomial regression models.
- Model performance evaluation using R-squared, Adjusted R-squared, Mean Squared Error (MSE), and Mean Absolute Error (MAE).

## Key Files

- `LRG_InsuranceDataset.ipynb`: The Jupyter notebook containing the full analysis and modeling steps.

## Data

The dataset used for this project includes the following variables:
- **Age**: Age of the policyholder.
- **BMI**: Body Mass Index.
- **Smoker Status**: Indicates whether the individual is a smoker.
- **Region**: Geographical region where the policyholder resides.
- **Children**: Number of children covered by health insurance.

## Methodology

1. **Data Preprocessing**: Handling missing data, encoding categorical variables, and scaling numeric features.
2. **Exploratory Data Analysis (EDA)**: Visualizing relationships between variables such as age, BMI, and insurance premiums.
3. **Linear Regression**: Building a linear model to predict premiums based on key features.
4. **Polynomial Regression**: Enhancing the model with polynomial features to capture non-linear relationships between age, BMI, and insurance premiums.
5. **Model Evaluation**: Using metrics such as R-squared, Adjusted R-squared, MSE, and MAE to evaluate model performance.

## Results

- The **linear regression model** achieved an R-squared of X.XXXX, indicating that it explains XX% of the variance in insurance premiums.
- The **polynomial regression model** achieved an R-squared of 0.7498, with an Adjusted R-squared of 0.7487.
- Mean Squared Error (MSE) of the polynomial model: 36,666,102.68.
- Mean Absolute Error (MAE) of the polynomial model: 4,262.26.

## Conclusion

The analysis reveals that smoker status and BMI are strong predictors of insurance premiums. The polynomial regression model slightly improves the fit over the linear model, suggesting some non-linear relationships between age, BMI, and premium costs. These insights can be leveraged to adjust pricing strategies based on demographic and lifestyle factors.

## Dependencies

- Python 3.12.4
- NumPy
- Pandas
- Statsmodels
- Scikit-learn
- Matplotlib

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/insurance-premium-analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd insurance-premium-analysis
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the Jupyter notebook to run the analysis:
   ```bash
   jupyter notebook ECON6485_A3_JaylenSheard.ipynb
   ```