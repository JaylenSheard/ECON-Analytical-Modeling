# KNN Analysis on Iris Dataset Findings

This repository contains the findings and analysis of applying the K-Nearest Neighbors (KNN) algorithm to the Iris dataset.

## Table of Contents
- [Data Preprocessing](#data-preprocessing)
- [Model Selection and Training](#model-selection-and-training)
- [Cross-Validation Results](#cross-validation-results)
- [Confusion Matrix Analysis](#confusion-matrix-analysis)
- [Model Robustness and Stability](#model-robustness-and-stability)
- [Potential Areas for Improvement](#potential-areas-for-improvement)
- [Overall Conclusion](#overall-conclusion)

## Data Preprocessing

- The Iris dataset was successfully loaded and split into training, validation, and test sets.
- The model was tested both with and without scaling to compare performance.

## Model Selection and Training

- Initial implementation: KNN model with `n_neighbors=3`
- A loop was created to compare the model's performance using various k values (1, 5, 7, and 9).

## Cross-Validation Results

5-fold cross-validation was used to evaluate the model's performance:

- Scores: [0.9667, 0.9667, 0.9333, 0.9, 1.0]
- Mean Accuracy: 95%
- Standard Deviation: 3%

**Insight**: The high mean accuracy and low standard deviation suggest that the KNN model performs consistently well on the Iris dataset, with minimal variance in performance across different folds.

## Confusion Matrix Analysis

The confusion matrix revealed:

- Class 0: Perfectly classified with no misclassifications.
- Class 1: 6 instances correctly classified, with 1 instance of Class 2 being incorrectly predicted as Class 1.
- Class 2: 3 instances correctly classified, with 1 instance misclassified as Class 1.

**Insight**: There is a slight confusion between Class 1 and Class 2, indicating that the model sometimes struggles to distinguish between these two classes. This confusion is expected as Iris Versicolor and Iris Virginica species have similar measurements for petal and sepal dimensions, causing overlap in the feature space.

## Model Robustness and Stability

The consistency of cross-validation scores and the low number of misclassifications suggest that the KNN model is both robust and stable for this dataset. The model is not overly sensitive to specific data splits.

## Potential Areas for Improvement

While the current model performs well, there is room for improvement in distinguishing between Class 1 and Class 2:

1. **Hyperparameter Tuning**: Experiment with different values of k to see if performance improves.
2. **Feature Engineering**: Introduce additional features or transform existing ones to enhance class separation.
3. **Other Algorithms**: Consider testing other classification models (e.g., Random Forests, Support Vector Machines) to see if they can better handle the slight overlap between classes.

## Overall Conclusion

The KNN model demonstrates strong performance on the Iris dataset, with a high overall accuracy and minor classification errors. Further tuning and exploration could optimize the model, especially in cases where precision in distinguishing between similar classes is critical.
