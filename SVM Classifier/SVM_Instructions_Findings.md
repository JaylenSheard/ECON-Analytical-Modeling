# Customer Churn Prediction using Support Vector Classifier (SVC)

## Project Overview

This project aims to predict customer churn using a Support Vector Classifier (SVC) on a customer churn dataset. The firm’s goal is to identify customers who are likely to churn and offer a retention incentive to maximize earnings. By predicting churn effectively, the firm can minimize revenue losses by providing a discount to at-risk customers and retaining them.

The project explores:

- **Data Preprocessing**: Cleaning and transforming the dataset.
- **Model Training**: Building a Support Vector Machine model with hyperparameter tuning using GridSearchCV.
- **Threshold Adjustment**: Optimizing the decision threshold for improved classification performance.
- **Business Impact Analysis**: Calculating total and average earnings per customer based on model predictions.
- **Confusion Matrix**: Evaluating the model's performance with visual aids.

## Dataset

The dataset used contains customer information relevant to churn, including features such as:

- Customer demographics
- Subscription details
- Usage patterns

The target variable is `Churn`:
- `1`: Customer will churn
- `0`: Customer will not churn

## Business Context and Earnings Calculation

The firm earns revenue based on the following logic:

- **True Negative (TN)**: The customer is predicted to stay, and indeed stays. Earnings: **$15**.
- **True Positive (TP)**: The customer is predicted to leave and offered a discount, but decides to stay. Earnings: **$10**.
- **False Negative (FN)**: The customer is predicted to stay, but actually leaves. Earnings: **$0**.
- **False Positive (FP)**: The customer is predicted to leave, but stays without any need for a discount. The firm loses **$5**.

This earnings model is implemented in the code to evaluate the classifier's performance beyond standard metrics (precision, recall, F1-score).

## Model Training and Hyperparameter Tuning

The model is a Support Vector Classifier (SVC) using an RBF kernel. Hyperparameter tuning is performed using **GridSearchCV**, where the best values of `C` and `gamma` are identified by maximizing the firm’s total earnings.

### Key Hyperparameters Tuned:

- **C**: Regularization parameter
- **Gamma**: Kernel coefficient for RBF kernel

## Custom Decision Threshold

Initially, the model performance is evaluated using standard metrics (precision, recall, F1 score) at the default threshold. A custom decision threshold was applied to maximize the precision and recall balance, enhancing the model's ability to classify potential churners effectively.

The custom threshold in this project is **-0.011**, which leads to improved performance metrics:
- **Precision**: 0.91
- **Recall**: 1.0
- **F1-score**: 0.95

## Performance Metrics

- **Initial Performance (without threshold tuning)**:
  - Precision: 1.0
  - Recall: 0.6
  - F1 Score: 0.749
  - ROC-AUC: 0.98
  
- **Final Performance (with threshold tuning)**:
  - Precision: 0.91
  - Recall: 1.0
  - F1 Score: 0.95

## Confusion Matrix

The performance of the model on the validation set is visualized using a confusion matrix. The confusion matrix provides insights into how well the model distinguishes between churners and non-churners.

Sample confusion matrix (before threshold tuning):
```
Predicted  0   1
Actual      
    0      9   1
    1      2   8
```

## Model Earnings Calculation

The model’s performance is evaluated based on the firm's earnings. The model selects the best combination of hyperparameters by maximizing the following earnings:

- **True Negative**: $15
- **True Positive**: $10
- **False Negative**: $0
- **False Positive**: -$5

The total and average earnings are calculated on both validation and test sets.

Sample output:
```
Total Earnings: 95
Average Earnings per Customer: 9.5
```

## Visualizations

- **Confusion Matrix**: A heatmap of the confusion matrix shows the performance on the validation set.
- **ROC Curve**: The model’s ability to distinguish between classes is assessed using the ROC curve.
- **Earnings Plot**: The earnings based on the confusion matrix for each iteration of hyperparameter tuning are tracked.

## How to Run the Project

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Load the dataset and split it into training and test sets:
   ```python
   from sklearn.model_selection import train_test_split
   x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

3. Standardize the features:
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   x_train_scaled = scaler.fit_transform(x_train)
   x_test_scaled = scaler.transform(x_test)
   ```

4. Run hyperparameter tuning using GridSearchCV:
   ```python
   param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']}
   grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
   grid.fit(x_train_scaled, y_train)
   ```

5. Apply the best model to the test set and calculate total earnings:
   ```python
   best_model = grid.best_estimator_
   y_pred = best_model.predict(x_test_scaled)
   cm = confusion_matrix(y_test, y_pred)
   test_earnings = calculate_earnings(cm)
   ```

## Future Enhancements

- **Feature Engineering**: Additional features such as customer engagement, length of subscription, and customer support interactions could enhance model accuracy.
- **Other Models**: Comparing performance with other classifiers like Random Forest, Gradient Boosting, or Neural Networks.
- **Advanced Tuning**: Explore advanced techniques like RandomizedSearchCV or Bayesian Optimization for hyperparameter tuning.

## Conclusion

This project successfully uses a Support Vector Classifier to predict customer churn and optimize business earnings by retaining at-risk customers. By tuning the decision threshold and maximizing firm-specific earnings, the model goes beyond traditional performance metrics to deliver business value.