##This file contains our machine learning model. The goal is to predict the best crop to plant (output) based on the weather conditions at the user's location (input).

##This machine learning model was coded with the help of:
## - Friends and family: help understanding basic ml concepts, help creating a structure for our code, advice on what algorithms to test/use, help finding suitable functions or code snippets to use for what we wanted to implement, corrections, help with fixing errors and problems, proof reading and so on.
## - CS Coaching tutors: advice on dataset quality and usability, overall feedback.
## - AI (Github Copilot, ChatGPT and Claude): help understanding basic ml concepts, help finding suitable functions to use for what we wanted to implement, help with errors and problems, corrections, help finding structures of blocks for what we wanted to implement, providing improvements and so on.


import pandas as pd         ##Used for working with DataFrames and more generally manipulating/analyzing data
import matplotlib.pyplot as plt         ##Used for creating plots (visual "graphs")
from pandas.plotting import scatter_matrix      ##Used for creating scatter plot matrix: visualizing relationships between variables
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score      ##Used to split data (training and validation sets), for cross validation, for model evaluation using cross validation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report         ##Used to measure accuracy of predictions, to show correct/incorrect predictions, to display the evaluation(scores)
from sklearn.linear_model import LogisticRegression         ##Used for logistic regression classification
from sklearn.tree import DecisionTreeClassifier         ##Used for classification with decision trees
from sklearn.neighbors import KNeighborsClassifier      ##Used for K-nearest neighbors classification (works by classifying data based on the closest training examples)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis        ##Used for linear discriminant analysis classification
from sklearn.naive_bayes import GaussianNB      ##Used for Gaussian classification
from sklearn.svm import SVC         ##Used for creating a ML model based on svm ("support vector machines") for classification
from sklearn.ensemble import RandomForestClassifier         ##Used for random forest classification
from sklearn.model_selection import GridSearchCV        ##Used for finding the best parameters for the model
import pickle       ##Used for saving the model to/from files


##Loading the dataset
url = 'Crop_Recommendation.csv'
names = ['Rainfall', 'pH_Value', 'Potassium', 'Temperature', 'Humidity', 'Crop']
df = pd.read_csv(url, encoding='cp1252')        ##Reading the CSV file. "cp1252" is a used character encoding (method to represent characters in binary data form, 0 and 1).

print(df.dtypes)        ##Printing data types of each column in df (dataframe), meaning that we can see the type of data in each column (e.g. int, float, and so on.).


##Checking if and which columns can be used for plotting (plots, "graphs")
numeric_cols = df.select_dtypes(include=['number', 'datetime']).columns         ##Selecting each column in df that has numeric or datetime data types.
if numeric_cols.empty:      ##If there's no columns with numeric or daytime data types found, it gives the error message.
    raise ValueError("The dataset does not contain any numeric or datetime columns for plotting.")
else:       ##If there are:
    print(f"Numeric columns for plotting: {numeric_cols}")      ##Printing the list of columns to show which ones can be used for plotting.


##Cleaning up the missing values and preparing the data: defining the input and output variables
df = df.dropna(subset=['Temperature', 'Humidity', 'Crop'])      ##"Cleaning up", meaning that we're dropping rows from df where the columns (Temperature, Humidity and Crop) have missing values (NaN). The df only keeps rows with full columns (where all 3 columns have data).

X = df[['Temperature', 'Humidity']].values      ##X is a numpy array. Temperature and Humidity columns from df are converted to X. X is used as the input for the machine learning model.
y = df['Crop'].values       ##y is a numpy array. Crop column from df is converted to y. y is used as the value to predict (crop recommendation) for the machine learning model.


##Counting values
print(df['Crop'].value_counts())        ##Counting how many times each value (each crop, e.g. rice, kidney beans, and so on.) appears in the Crop column of df. We do this to check the distribution of data.


##Creating and showing histographs (distribution of data)
df.hist()       ##Using matplotlib, creating histographs, which basically serve to show the distribution of data in the dataset, for each numeric column in df.
plt.show()      ##Showing the histographs (distribution of data). (Opening up a plot window to display them).


##Creating and showing scatter plot matrix
scatter_matrix(df)      ##Creating a scatter plot matrix which shows the relationship between variables.
plt.show()      ##Showing the scatter plot matrix

# # Histograms for Temperature and Humidity
# df[['Temperature', 'Humidity']].hist()
# plt.show()
#
# # Scatter plot matrix for Temperature and Humidity
# scatter_matrix(df[['Temperature', 'Humidity']])
# plt.show()

##Splitting data into training and validation sets
##Using scikit-learn function "train_test_split" to split/divide the dataset. Input data (X) and output data (y) are split, they go into two groups: 80% (0.8) for training (X_train, y_train) and 20% (0.2) for validation (X_val, y_val).
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)      ##"test_size=0.2" so that 20% of data goes to validation, "random_state=1" to ensure that the split is the same every time we run the code, "stratify=y" to make sure that there's the same proportion of the classes in the sets (as the goal is to have balanced classification).


##Parameter grid: creating a dictionary ("param_grid") to list values for different parameters of the model.
param_grid = {
    'n_estimators': [50, 100, 200],         #Number of trees in the forest
    'max_depth': [None, 10, 20, 30],        ##Maximum depth of the tree (how many splits each tree can have).
    'min_samples_split': [2, 5, 10],        ##Minimum number of samples required to split an internal node (to be able to split a node).
    'min_samples_leaf': [1, 2, 4],      ##Minimum number of samples required to be at a leaf node (to avoid having leaf nodes with too few samples).
    'bootstrap': [True, False]      ##If true, each tree is trained on a random sample of data with replacement, if false, all trees are trained on the same data.
}


##Creating a learning model: Random Forest, for classification
rf = RandomForestClassifier(random_state=1)         ##RandomForestClassifier (scikit-learn) to build multiple decision trees and combine their results to give predictions.
##"random_state=1" to make sure that the results are the same every time we run the code.


##Creating a GridSearch Cv, meaning that we are trying to test different combinations of parameters from "param_grid" to find the best one. Accuracy is used as metric.
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)       #Running the grid search on the training data (X_train, y_train) to find the best combination of parameters from "param_grid".
## "cv=5" means that we are using 5-fold cross-validation (splitting the data into 5 parts and training on 4 parts and validating on 1 part, repeating this process 5 times). "verbose=2" to see detailed output of the process. "n_jobs=-1" to speed up the process by using all available CPU cores.


##Best parameters and accuracy score
print("Best Parameters:", grid_search.best_params_)         ##Displaying best set of parameters that was found.
print("Best Accuracy:", grid_search.best_score_)        ##Displaying best cross-validated accuracy score.


##Saving the best model found (optional)
best_rf = grid_search.best_estimator_


##List of ML algorithms to compare or evaluate which one works best on our data.
models = [
    ('LR', LogisticRegression(solver='liblinear')),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto')),
    ('RF', RandomForestClassifier(random_state=1))
]


##"results[]" and "names[]": creating empty lists to store results and names of the models
results = []
names = []
print("Évaluation des modèles:")        ##Evaluating models
for name, model in models:      ##Using "for" loop to go through each model in the "models" list
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)      ##Cross validation: "n_splits=10": splitting data into 10 parts, "shuffle=True": shuffling randomly the data before splitting, "random_state=1": make sure that the results are the same every time we run the code.
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')         ##Evaluating model with cross validation ("cv") and returning accuracy score each time.
    results.append(cv_results)      ##Adding the accuracy score for the model to the "results" list.
    names.append(name)      ##Adding the name of the model to the "names" list.
    print(f'{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})')      ##Printing the name of the model, accuracy score (mean) and the standard deviation (std) of accuracy score to see score variations.

plt.boxplot(results, tick_labels=names)         ##Creating a boxplot to show visually the accuracy score distribution for each model.
plt.title('Comparison of algorithms')       ##Title of the boxplot
plt.show()      ##Showing the boxplot


##Creating the final model: RandomForestClassifier with the following settings
final_model = RandomForestClassifier(
    bootstrap=True,         ##Each tree is trained on a random sample of data (with replacement). If false, all trees would have been trained on the same data.
    max_depth=None,         ##Gives limit to maximum how many splits each tree can have. We're not using a limit ("None").
    min_samples_leaf=4,         ##Gives the minimum number of samples to be at a leaf node (avoiding having leaf nodes with too few samples).
    min_samples_split=2,        ##Gives the minimum number of samples to be able to split a node.
    n_estimators=200,       ##Number of trees (influences the performance of the model and the time it needs).
    random_state=1      ##To make sure that the results are the same every time we run the code.
)


##Training the final ML model
final_model.fit(X_train, y_train)       ##"final_model" is the final model used, "X_train" and "y_train" are the training data ("X_train" as input and "y_train" as output).


##Making predictions on the validation data and storing results in "predictions". Overall, checking the performance of the model (final model).
predictions = final_model.predict(X_val)        ##"X_val" is the validation data (input).


##Showing an "evaluation" of the model
print('Accuracy:', accuracy_score(y_val, predictions))      ##Giving the accuracy of the model (the proportion of correct predictions or outputs).
print('Confusion matrix:\n', confusion_matrix(y_val, predictions))      ##Giving the confusion matrix to show how many correct and incorrect predictions we have.
print('Classification report:\n', classification_report(y_val, predictions))        ##Giving the classification report to show the precision, recall and F1-score for each class (each crop). Overall, we used it to show the global performance of the model.


##Opening the file "crop_rf_model.pkl"
with open('crop_rf_model.pkl', 'wb') as f:      ##"wb" = "write binary", writing in binary mode.
    pickle.dump(final_model, f)         ##Using pickle to save the final model. ("f" for file, it's just to show pickle.dump() where the model is saved).