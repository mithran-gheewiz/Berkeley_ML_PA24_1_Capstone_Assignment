# Berkeley_ML_PA24_1_Capstone_Assignment
# Link to Python notebooks: 

## Introduction and Problem Statement
The problem that I am intending to solve is to build a predictive model that can determine if parts in a sequence of manufacturing operations of a leading Automotive Tier-1 supplier through multiple stations can be deemed a good part or a defective part.  I intend to address the question of whether the precision, recall, AUC-ROC, and F1-score between the predicted and observed responses falls within an acceptable range for the test dataset. These metrics are performance metrics that takes into account true and false positives and negatives, and is especially useful for imbalanced classification problems.

The data set is from the Bosch Kaggle competition. The data is sourced from Kaggle. https://www.kaggle.com/competitions/bosch-production-line-performance/overview

The dataset is huge containing three types of feature data: numerical, categorical, date stamps and the labels indicating the part as good or bad. The training data has 1,184,687 samples and the learned model will be used to predict on a test dataset containing 1,183,748 samples. There are 968 numerical features, 2140 categorical features and 1156 date features. I used a limited data set of randomized 100,000 rows from the training datasets and randomized 30,000 rows from the test datasets.


## Model Outcomes or Predictions

In the first part of the Capstone Project, as a baseline model, I used Logistic regression model - L1 regularization and Decision tree model (Tree depth (n=5))
with Hyper parameter tuning for both logistic regression and decision tree. 

Table 2 shows the results of the decision tree and the logistic regression using the best F1th and the baseline.

                                Table 1 Baseline Model Evaluation
                                Both baseline models were completely ineffective at detecting defects. 
                                
<img width="808" height="221" alt="image" src="https://github.com/user-attachments/assets/267f969b-6530-4f65-b941-0be445ca9368" />


Logistic L1 is a completely ineffective model here. It collapses to predicting only the majority class. Low RMSE (0.0756) but useless in terms of classification metrics (especially recall/F1). The model is predicting all negatives (never predicts “1”). Since there are true positives in the data (115 positives), this means it completely missed them. Hence Recall = 0 and Precision = 0. Therefore, the logistic regression L1 cannot be used to make any predictions.

For the decision tree model, by moving the threshold up to 0.9, the tree becomes more conservative (predicts fewer positives). This improves precision slightly but sacrifices recall. F1 is still very poor (0.09), but it beats the other setup.

                      Table 2. Model Outcomes for Different models used (VotingClassifier, XGBoost, and RandomForest)
                      The best model surprisingly is VotingClassifier (SOFT, best-F1=0.70) compared to the other Ensemble Models 
| Model / Threshold Setting                 | AUC-ROC | Precision | Recall | F1 Score |
| ----------------------------------------- | ------- | --------- | ------ | -------- |
| **VotingClassifier (HARD)**               | 0.6680  | 0.0000    | 0.0000 | 0.0000   |
| **VotingClassifier (SOFT, 0.50)**         | 0.6680  | 0.0293    | 0.2743 | 0.0530   |
| **VotingClassifier (SOFT, best-F1=0.70)** | 0.6680  | 0.1176    | 0.2301 | 0.1557   |
| **XGBoost (0.50)**                        | 0.5753  | 0.2667    | 0.0331 | 0.0588   |
| **XGBoost (best-F1=0.60)**                | 0.5753  | 0.3333    | 0.0331 | 0.0602   |
| **Random Forest (0.50)**                  | 0.5837  | 0.0000    | 0.0000 | 0.0000   |
| **Random Forest (best-F1=0.24)**          | 0.5837  | 0.0274    | 0.0744 | 0.0401   |

VotingClassifier (SOFT, best-F1 threshold=0.70) performed best overall, achieving the highest F1 score (0.1557). The confusion matrix shows [TN=19674, FP=195, FN=87, TP=26].
This means it successfully caught more true positives compared to other models, while keeping false positives relatively low.
Although the recall (0.23) is not high, it’s meaningfully better than XGBoost and RandomForest.

XGBoost had higher precision (0.3333 at threshold=0.60), but recall was very low (0.0331).
Its confusion matrix [TN=19856, FP=8, FN=117, TP=4] shows it rarely misclassified negatives (very few FPs), but it missed almost all positives.
This makes XGBoost too conservative — it predicts almost everything as negative, leading to low recall.

Random Forest underperformed in both precision and recall.  Even at its best threshold (0.24), the F1 score was only 0.0401.
Its confusion matrix [TN=19545, FP=319, FN=112, TP=9] shows it allowed more false positives than XGBoost but still failed to capture enough true positives.
This means that Random Forest is absolutely not the choice for deployment even though it is a powerful Ensemble model. The reasons for this will be discussed in the Model Evaluation section. 


## Data Acquisition
Using random sampling, I was able to get the following rows from the training data sets:
train_numerical: (99922, 970), First ID: 49 train_categorical: (99922, 2141), First ID: 49 train_date: (99922, 1157), First ID: 49

The data checks shows that there is a large number of missing values in the sampled 100,000 data frame for all three files (numerical, categorical, and date). This will pose a significant issue for logistic regression, which is sensitive to NaN or missing values, and also may pose an overfit condition for XGBoost and RandomForest. A complete discussion on the models used is in the Model Outcomes or Predictions section and the Model Evaluation sections. 

Before data processing I must drop rows that have lots of NaNs, and/or use PCA and/or L1 regression that will reduce factors that are not important to zero. In addition, I used Imputers to clean missing values. Here is an example of the average number of missing values: Average NaNs per row numerical: 784.98 Average NaNs per row categorical: 2082.95 Average NaNs per row date: 950.86

Similarly, I also randomly sampled 30,000 rows from the test data sets with the following results: test_numerical: (30001, 969), First ID: 19 test_categorical: (30001, 2141), First ID: 19 test_date: (30001, 1157), First ID: 19

## Data Preprocessing/Preparation

## Modeling
### Baseline Modeling - Logistic Regression and Decision Tree
In the first part of the Capstone Project, I use the following baseline models in the analysis:

Logistic regression model - L1 regularization
Decision tree model (Tree depth (n=5))
Hyper parameter tuning for both logistic regression and decision tree. 

When I first started out, the baseline model did not converge. Then, I performed dimensionality reduction before the classifier (often best for OHE-heavy data) and reran the program and its still running after 12 hours without completing the kernal. I determined that the bottlenecks are: refitting OHE+SVD inside every CV fold, too many candidates/folds, and using a heavy solver on high-dim data. To make it run faster, I did the following: Tune on a stratified subset (30k rows), then refit best params on the full 100k. Use HalvingRandomSearchCV (successive halving) with 3-fold CV. Shrink SVD to ~50 comps and n_iter=2. Use L1 + liblinear (binary) after SVD (which is way faster than saga here). For the tree, I compressed the categorical branch with OHE to SVD inside the ColumnTransformer (dramatically fewer features).

The above actions allowed the models to converge and provide outputs on precision, recall, and F1-scores. The discussion of the baseline modeling will be done in Model Evaluation section.

### Modeling subsection on VotingClassifier, XGBoost, RandomForest
Based on the Module 20 lesson on Voting Classifier, I determined if adding a Voting Classifier will help improve the F1 and precision/recall scores. I investigated both Hard and Soft voting. Since decision tree performed better than logistic regression, I decided to weight it 3:1 in favor of decision tree. By combining two weaks models, I am hoping to get better results. The voting classifier models included both hard and soft voting.

Pipelines inside the voter: best_logit and best_tree already encapsulate imputation, scaling/OHE, SVD. Ensembling them directly keeps preprocessing consistent and prevents leakage.
Soft voting + thresholding: With extreme class imbalance, probability averaging plus an optimized decision threshold typically beats hard voting at F1/recall. After changing weights, I retuned the decision threshold on X_val (0.5 to 0.7 for best threshold) to squeeze out more from F1/recall/precision. 

This was followed by using the ensemble method XGBoost to see if the performance is better. It uses a preprocessing path for trees (no scaling), compresses high-cardinality OHE with SVD, tunes a small hyperparam space with HalvingRandomSearchCV, and then evaluates with both 0.5 and best-F1 thresholds. I had to first install XGBoost using the PIP install command. 
Bosch categorical has huge cardinality; compressing OHE into ~100 components keeps XGBoost fast and memory-safe while still capturing signal. 
I evaluated XGBoost on the same hold-out and with the same thresholding strategy used for Logistic Regression/ Decision Tree/VoterClassfier, then sort by F1 to determine the best performing model.

After discussing with my learning facilitator, I built a model using Random Forest to compare the results with XGBoost, Voting Classifier and the other baseline models. I trimmed search space, uses HalvingRandomSearchCV, which lowered SVD size, increased category bucketing, and parallelized trees. The Random Forest pipeline matches the previous setups (median-impute nums, OHE to SVD for categoricals), does a wider RandomizedSearchCV on the 30k tuning subset, then evaluates on the validation split with both 0.5 and best-F1 thresholds.

## Model Evaluation

### Logistic Regression and Decision Tree does not make the cut
Logistic L1 is a completely ineffective model here. It collapses to predicting only the majority class. Low RMSE (0.0756) but useless in terms of classification metrics (especially recall/F1). The model is predicting all negatives (never predicts “1”). Since there are true positives in the data (115 positives), this means it completely missed them. Hence Recall = 0 and Precision = 0. Therefore, the logistic regression L1 cannot be used to make any predictions.

For the decision tree model, by moving the threshold up to 0.9, the tree becomes more conservative (predicts fewer positives). This improves precision slightly but sacrifices recall. F1 is still very poor (0.09), but it beats Logistic Regression. This model also cannot be used to make any useful predictions on defective parts.

### Precision, Recall Curves and AUC-ROC Curves for Voting Classifier, XGBoost and Random Forest

        Fig. 1. Confusion Matrix of the models showing that the Voting Classifier (SOFT threshold, @ 0.70) performed the best by catching more actual positives (26 true positives) than XGBoost or Random Forest.
        And, keeps false positives manageable (195 out of ~20k).
       
        
  <img width="1105" height="828" alt="image" src="https://github.com/user-attachments/assets/a8615019-0a2c-4c15-b4e4-bc14ab4e1f3c" />

- VotingClassifier (SOFT @ 0.70) captures more true positives (top-right) while keeping false positives moderate.
- XGBoost almost never predicts positives (tiny TP count, very few FPs).
- Random Forest struggles, either predicting none (0.50) or adding many false positives without enough true positives (0.24).
This visualization reinforces why the VotingClassifier (SOFT, best-F1) is the strongest option.


        Fig. 2. ROC curve comparison for VotingClassifier (SOFT), XGBoost, and Random Forest. VotingClassifier (SOFT) clearly sits above the other two curves, reflecting its higher AUC (0.668).

<img width="779" height="539" alt="image" src="https://github.com/user-attachments/assets/237bfb49-1a60-43c2-9fc1-05e29ce7648d" />


- VotingClassifier (SOFT) clearly sits above the other two curves, reflecting its higher AUC (0.668).
- XGBoost (0.575) and Random Forest (0.584) are only slightly better than random guessing (AUC = 0.5).

This shows why the VotingClassifier gave a better trade-off between sensitivity (recall - 0.23, versus XGBoost recall - 0.03, and Random Forest recall - 0.07) and specificity, and why threshold tuning helped it outperform the others.



      Fig. 3. ROC curve comparison again — with colored dots showing where each model’s chosen threshold lands. VotingClassifier (SOFT @ 0.70) sits in a better region: higher recall than XGBoost/RandomForest
      
<img width="1113" height="619" alt="image" src="https://github.com/user-attachments/assets/0990cab9-3b4d-41ce-a8ba-358782a8a204" />


- VotingClassifier (SOFT @ 0.70) sits in a better region: higher recall than XGBoost/RandomForest with moderate false positives.
- XGBoost points are way down near the x-axis — almost zero recall, which explains why it missed nearly all positives.
- Random Forest also clusters near the baseline with very low recall.
- VotingClassifier (SOFT @ 0.50) has higher recall but at the cost of more false positives.

This visual confirms that only the VotingClassifier (SOFT) provided usable trade-offs on the ROC curve, while the other ensembles defaulted to predicting negatives too aggressively.

      Fig. 4. Precision-Recall curve (operating points). VotingClassifier (SOFT @ 0.70) is the best with Precision ~0.12 with Recall ~0.23 — better balance than the others.
<img width="987" height="534" alt="image" src="https://github.com/user-attachments/assets/538fb333-9166-4924-b4b5-3b9c5ca03ab8" />

- VotingClassifier (SOFT @ 0.70) again stands out:
- Precision ~0.12 with Recall ~0.23 — better balance than the others.
- XGBoost points are high precision but very low recall (bottom-left corner) → it only catches a handful of positives.
- Random Forest sits near the baseline, barely above random guessing.
- VotingClassifier (SOFT @ 0.50) trades higher recall (~0.27) for very low precision (~0.03).


### Why the VotingClassifier (SOFT) Outperformed XGBoost and Random Forest

One of the main reasons the VotingClassifier (SOFT) outperformed XGBoost and Random Forest is the issue of class imbalance. In the dataset, the number of positive cases is much smaller than the number of negatives. Both Random Forest and XGBoost tend to optimize for accuracy and AUC, which often means predicting “negative” most of the time. This strategy minimizes errors overall but comes at the cost of missing true positives. As a result, their confusion matrices show very few false positives but also almost no true positives, making them overly conservative. For example, XGBoost at threshold 0.60 produced a confusion matrix of [TN=19856, FP=8, FN=117, TP=4], which means it only detected 4 true positives. Precision was relatively high, but recall collapsed to almost zero.

Another advantage of the VotingClassifier (SOFT) lies in probability averaging. By combining Logistic Regression and Decision Tree models with a 1:3 weight, the soft voter averages their probability outputs. Logistic regression contributes smoother, more continuous probability estimates, while decision trees provide sharper splits that can highlight specific patterns. When averaged together, the resulting probability distribution is less extreme than those produced by XGBoost or Random Forest. This makes threshold tuning more effective. At a threshold of 0.70, the soft voter identified 26 true positives with a reasonable precision level, striking a better balance between precision and recall than the other methods.

A further reason is the contrast between overfitting and simplicity. XGBoost and Random Forest are highly flexible models, which makes them powerful but also prone to overfitting, especially when data is noisy or features are sparse. In cases where the true signal is weak — as often happens with imbalanced datasets — these models may fail to generalize and instead fall back on predicting negatives. In contrast, Logistic Regression and Decision Tree, while simpler, are complementary in nature. Together in a soft voter, they can be more robust and capture meaningful signals that the more complex models overlook.

Finally, threshold tuning played a key role. The performance of the VotingClassifier improved substantially when the threshold was adjusted from the default 0.50 to the best-F1 threshold of 0.70. This adjustment helped the model capture more positives without overwhelming the system with false positives. XGBoost and Random Forest did not benefit as much from threshold tuning because their predicted probabilities were already heavily skewed toward negatives, leaving little flexibility.

In summary, while XGBoost and Random Forest are generally strong learners, in this particular imbalanced setting they behaved like “all negative” classifiers — protecting precision but sacrificing recall. The VotingClassifier (SOFT), by averaging two weaker but diverse models, created a smoother probability landscape. This allowed for better threshold adjustments, enabling it to capture a meaningful number of positives while keeping false positives under control.

## Business Impact

Manufacturing companies pay particular attention to defective parts to ensure that these parts do not land in customers' hands. The ramifications of defective parts going out of a factory can be significant to any company that manufactures parts. These include costly product recall, loss of reputation and loss of future business.
Since the business objective is to maximize defect detection (catching faulty/defective parts), VotingClassifier (SOFT, 0.70) is clearly superior as discussed in the modeling evaluation section. 

In a different business setting, if the business objective are to minimize false alarms at all costs, XGBoost could be considered — but the trade-off is that nearly all positives would be missed, which is unacceptable in quality control or risk detection scenarios.

Therefore the recommendation is to deploy the VotingClassifier (SOFT, 0.70) model which will be best at catching defective parts, has the best trade off between Precision ~0.12 with Recall ~0.23, and has the best F1 Score ~0.156. Compared to the other models, this model will likely detect 23% of the defects versus the other instead only detect less than 5%. The model VotingClassifier (SOFT, 0.70)  also performs better than random chance like the other ensemble models with an AUC score of 0.668.

