# Berkeley_ML_PA24_1_Capstone_Assignment
# Link to Python notebooks: 

## Introduction and Problem Statement
The problem that I am intending to solve is to build a predictive model that can determine if Bosch parts in a sequence of manufacturing operations through multiple stations can be deemed a good part or a defective part.  I intend to address the question of whether the precision, recall, AUC-ROC, and F1-score between the predicted and observed responses falls within an acceptable range for the test dataset. These metrics are performance metrics that takes into account true and false positives and negatives, and is especially useful for imbalanced classification problems.

The data set is from the Bosch Kaggle competition. The data is sourced from Kaggle. https://www.kaggle.com/competitions/bosch-production-line-performance/overview

The dataset is huge containing three types of feature data: numerical, categorical, date stamps and the labels indicating the part as good or bad. The training data has 1,184,687 samples and the learned model will be used to predict on a test dataset containing 1,183,748 samples. There are 968 numerical features, 2140 categorical features and 1156 date features. I used a limited data set of randomized 100,000 rows from the training datasets and randomized 30,000 rows from the test datasets.

Using random sampling, I was able to get the following rows from the training data sets:
train_numerical: (99922, 970), First ID: 49 train_categorical: (99922, 2141), First ID: 49 train_date: (99922, 1157), First ID: 49

The data checks shows that there is a large number of missing values in the sampled 100,000 data frame for all three files (numerical, categorical, and date). This will pose a significant issue for logistic regression, XGBoost and RandomForest which is sensitive to NaN or missing values. A complete discussion on the models used is in the Model Outcomes or Predictions section. Must drop rows that have lots of NaNs, and/or use PCA and/or L1 regression that will reduce factors that are not important to zero. Here is an example of the average number of missing values: Average NaNs per row numerical: 784.98 Average NaNs per row categorical: 2082.95 Average NaNs per row date: 950.86

Similarly, I also randomly sampled 30,000 rows from the test data sets with the following results: test_numerical: (30001, 969), First ID: 19 test_categorical: (30001, 2141), First ID: 19 test_date: (30001, 1157), First ID: 19

## Model Outcomes or Predictions

### Table 1. Model Outcomes for Different models used (VotingClassifier, XGBoost, and RandomForest)
### Caption: The best model surprisingly is VotingClassifier (SOFT, best-F1=0.70) compared it to the other Ensemble Models

| Model / Threshold Setting                 | AUC-ROC | Precision | Recall | F1 Score |
| ----------------------------------------- | ------- | --------- | ------ | -------- |
| **VotingClassifier (HARD)**               | 0.6680  | 0.0000    | 0.0000 | 0.0000   |
| **VotingClassifier (SOFT, 0.50)**         | 0.6680  | 0.0293    | 0.2743 | 0.0530   |
| **VotingClassifier (SOFT, best-F1=0.70)** | 0.6680  | 0.1176    | 0.2301 | 0.1557   |
| **XGBoost (0.50)**                        | 0.5753  | 0.2667    | 0.0331 | 0.0588   |
| **XGBoost (best-F1=0.60)**                | 0.5753  | 0.3333    | 0.0331 | 0.0602   |
| **Random Forest (0.50)**                  | 0.5837  | 0.0000    | 0.0000 | 0.0000   |
| **Random Forest (best-F1=0.24)**          | 0.5837  | 0.0274    | 0.0744 | 0.0401   |


## Data Acquisition

## Data Preprocessing/Preparation

## Modeling
In the first part of the Capstone Project, I use the following baseline models in the analysis:

Logistic regression model - L1 regularization
Decision tree model (Tree depth (n=5))
Hyper parameter tuning for both logistic regression and decision tree
When I first started out, the baseline model did not converge. Then, I performed dimensionality reduction before the classifier (often best for OHE-heavy data) and reran the program and its still running after 12 hours without completing the kernal. I determined that the bottlenecks are: (1) refitting OHE+SVD inside every CV fold, (2) too many candidates/folds, and (3) using a heavy solver on high-dim data. To make it run faster, I did the following: Tune on a stratified subset (30k rows), then refit best params on the full 100k. Use HalvingRandomSearchCV (successive halving) with 3-fold CV. Shrink SVD to ~50 comps and n_iter=2. Use L1 + liblinear (binary) after SVD (way faster than saga here). For the tree, compress the categorical branch with OHE → SVD inside the ColumnTransformer (dramatically fewer features).

The above actions allowed the models to converge and provide outputs on precision, recall, AUC-ROC, and F1-scores.  

## Model Evaluation
