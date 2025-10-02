# Berkeley_ML_PA24_1_Capstone_Assignment
# Link to Python notebooks: 

## Problem Statement
The problem that I am intending to solve is to build a predictive model that can determine if Bosch parts in a sequence of manufacturing operations through multiple stations can be deemed a good part or a defective part.  I intend to address the question of whether the precision, recall, AUC-ROC, and F1-score between the predicted and observed responses falls within an acceptable range for the test dataset. These metrics are performance metrics that takes into account true and false positives and negatives, and is especially useful for imbalanced classification problems.

The data set is from the Bosch Kaggle competition. The data is sourced from Kaggle. https://www.kaggle.com/competitions/bosch-production-line-performance/overview

The dataset is huge containing three types of feature data: numerical, categorical, date stamps and the labels indicating the part as good or bad. The training data has 1,184,687 samples and the learned model will be used to predict on a test dataset containing 1,183,748 samples. There are 968 numerical features, 2140 categorical features and 1156 date features. I used a limited data set of randomized 100,000 rows from the training datasets and randomized 30,000 rows from the test datasets.

Using random sampling, I was able to get the following rows from the training data sets:
train_numerical: (99922, 970), First ID: 49 train_categorical: (99922, 2141), First ID: 49 train_date: (99922, 1157), First ID: 49

The data checks shows that there is a large number of missing values in the sampled 100,000 data frame for all three files (numerical, categorical, and date). This will pose a significant issue for logistic regression, XGBoost and RandomForest which is sensitive to NaN or missing values. A complete discussion on the models used is in the Model Outcomes or Predictions section. Must drop rows that have lots of NaNs, and/or use PCA and/or L1 regression that will reduce factors that are not important to zero. Here is an example of the average number of missing values: Average NaNs per row numerical: 784.98 Average NaNs per row categorical: 2082.95 Average NaNs per row date: 950.86

Similarly, I also randomly sampled 30,000 rows from the test data sets with the following results: test_numerical: (30001, 969), First ID: 19 test_categorical: (30001, 2141), First ID: 19 test_date: (30001, 1157), First ID: 19

## Model Outcomes or Predictions

## Data Acquisition

## Data Preprocessing/Preparation

## Modeling

## Model Evaluation
