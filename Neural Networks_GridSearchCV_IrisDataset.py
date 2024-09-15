"""
This program uses GridSearchCV to find the best hyperparameters of MLPClassifier before training and testing on the Iris dataset.
First, the data is split into random and equal groups of training data and testing data.
Second, a list of hyperparameters for GridSearchCV to test is created.
Third, GridSearchCV goes through the list of hyperparameters and selects the ones that give a higher accuracy.
Fourth, the model is trained on the Iris dataset using the MLPClassifier which is adjusted based on the selection made with GridSearchCV.
Fifth, the model is tested on the testing data and the accuracy is printed.

This program generally has a high accuracy rate in the upper 90's.
If this were used on a dataset with words like the IMDB dataset, there would be added steps of pre-processing.
"""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as MLP

######
#DATA#
######
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size = 0.5, train_size = 0.5, shuffle = True, stratify = iris['target'])

###########
#ALGORITHM#
###########
parameters = {'hidden_layer_sizes':[(4, 5), (6, 8), (9, 12, 4)], 'learning_rate_init':[.01, .001, .0001], 'max_iter':[8000, 9000]}
clf = GridSearchCV(MLP(), parameters)

##########
#TRAINING#
##########
clf.fit(X_train, y_train)

####################
#TESTING/PREDICTION#
####################
y_pred = clf.predict(X_test)

print(f'The predicted classes are {y_pred}')
print(f'The best parameters are {clf.best_params_}')
print(f'The accuracy of the model is {accuracy_score(y_test, y_pred)*100}%')
