# Capstone-Project-Litigation-Prediction

Machine Learning model to predict the likelihood of Litigation
 
This is my capstone project with the company ESIS Inc.  We have received historical data for the past 5 years to make a model that will predict the likelihood of Litigation.
 
For the Data cleaning process, we dropped duplicates, handled null values, make data consistent by converting all text to lower and strip trailing white spaces merged few columns by mapping Boolean values to 0 and 1.
 
Then divided data into Training (60%), Validation (20%) and Test (20%) set. After exploratory data analysis, remove columns that are highly correlated with the column that we were predicting.
 
For Data modeling we use Decision Tree, Random Forest, Naïve Bayes, Logistic Regression, Gradient Boosting, and Multi-Layer Perceptron algorithms. We shortlist the models which giving a good F1 score (this metric was chosen because data was imbalanced). Then we did hyperparameter tuning for our selected models.
 
To handle imbalance data then we performed SMOTE to introduce synthesized data. After SMOTE, we stacked our model to improve the performance.

