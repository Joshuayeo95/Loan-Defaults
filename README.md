# Loan Defaults Modelling
-------------------------

In this project, the aim was to develop a model that would help to identify borrowers who were the most likely to default on their loans.

Preprocessing and Exploratory Data Analysis results can be viewed in the `eda.ipynb` notebook, and the training and validation of different models can be found in the `ml_models.ipynb` notebook.

These notebooks contain clear documentation of my thought process as i was working through the project.

__Note__: The functions used in this notebook can be found in the scripts within the utils folder.

## Machine Learning Models

* NearestCentroid
* K-Nearest Neighbours
* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier

## Resampling Techniques

From our exploratory data analysis, we note that we are working with an __imbalanced dataset__. Therefore, we will be applying different __resampling techniques__ and monitor if there are improvements to the model's performance.

1. Random Over Sampling
2. Synthetic Minority Oversampling Technique

## Data Preparation and Evaluation Metric

The dataset was split into training (__80%__) and holdout (__20%__) datasets.

The individual models were trained using __5-Fold cross validation__ on the training data before being evaluated on the holdout data.

The primary metric used to evaluate model performance is the __F1 Score__, which is the harmonic mean between precision and recall. The rationale for choosing F1 is that we want to __identify as many defaults as possible while ensuring that our model is not over predicting the defaults__. 

Additionally, we will provide insgihts about the model performances in relation to its __precision__ and __recall__ and how different use cases may favour different models.