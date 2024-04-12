## Machine Learning Algorithms Portfolio

This repository provides a concise overview of essential machine learning algorithms commonly used in various data science tasks. The code and data for these examples are located in the ml_portfolio: [ML Datasets](https://github.com/ericyoc/ml_portfolio/tree/main/ml_data) directory on GitHub.


## Motivating Article
Yu, L., Zhao, X., Huang, J., Hu, H., & Liu, B. (2023, December 29). Research on Machine Learning with Algorithms and Development. Journal of Theory and Practice of Engineering Science, 3(12), 7–14. https://doi.org/10.53469/jtpes.2023.03(12).02


## Machine Learning Algorithm Types

**Supervised Learning Algorithms**

*Multi-Linear Regression (MLR)**

**Summary:** Predicts a continuous output value based on a linear relationship with multiple input features.

**MLR Dataset:** FuelConsumptionCo2.csv

**Importance:** Simple to understand and interpret, efficient for linear data.

**Used for:** Sales forecasting, stock price prediction, risk assessment.

*Logistic Regression**

**Summary:** Classifies data points into discrete categories (binary or multi-class) using a sigmoid function.

**Logistic Regression Dataset:** ChurnData.csv

**Importance:** Widely used for classification problems, handles binary and multi-class scenarios.

**Used for:** Spam filtering, image classification, customer churn prediction.

*K-Nearest Neighbors (KNN)**

**Summary:** Classifies data points based on the majority vote of their k nearest neighbors in the training data.

**KNN Dataset:** teleCust1000t.csv

**Importance:** Easy to implement, performs well with high-dimensional data.

**Used for:** Image recognition, handwriting recognition, recommendation systems.

*Support Vector Machines (SVM)**

**Summary:** Creates a hyperplane that maximizes the margin between data points of different classes.

**SVM Dataset:** cell_samples.csv

**Importance:** Efficient with high-dimensional data, performs well with small datasets.

**Used for:** Text classification, image segmentation, anomaly detection.

*Regression Trees**

**Summary:** Tree-like models that make predictions based on a series of decision rules applied to input features.

**Regression Trees Dataset:** real_estate_data.csv

**Importance:** Easy to interpret, handles non-linear relationships well.

**Used for:** Customer segmentation, fraud detection, credit risk assessment.

*Decision Trees**

**Summary:** Tree-based models that use a series of decision rules to classify data points into different categories.

**Decision Trees Dataset:** drug200.csv

**Importance:** Easy to interpret, can handle both categorical and numerical data, and performs feature selection inherently.

**Used for:** Medical diagnosis, credit risk assessment, customer segmentation.

**Unsupervised Learning Algorithms**

*K-Means Clustering**

**Summary:** Groups data points into k clusters based on their similarity, often measured by distance.

**K-Means Clustering Dataset:** Cust_Segmentation.csv

**Importance:** Unsupervised learning for data exploration and segmentation.

**Used for:** Customer segmentation, market research, image compression.

**Deep Learning**

*Regression with Keras**

**Summary:** Keras, a deep learning framework, can be used to build a variety of neural network architectures for regression tasks, allowing for the modeling of complex relationships between input features and continuous outputs.

**Regression with Keras Dataset:** concrete_data.csv

**Importance:** Highly flexible and powerful for complex non-linear relationships.

**Used for:** Time series forecasting, image recognition, natural language processing.

**Note:** This repository does not include the Iris dataset for multi-class classification. You can find the Iris dataset from various online sources.

**Choosing the Right Algorithm**

The selection of an appropriate machine learning algorithm depends on several factors, including:

*Problem Type:** Supervised learning for prediction (classification or regression), Unsupervised learning for data exploration (clustering).

*Data Characteristics:** Linearity, dimensionality, presence of noise or outliers.

*Interpretability Needs:** Some algorithms offer clearer insights into the relationships between features and outputs.

*Computational Resources:** Some algorithms require more training time and computational resources than others.

By understanding the strengths and limitations of each algorithm, you can make an informed decision when tackling your specific machine learning problem.
