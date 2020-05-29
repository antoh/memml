# import modules
import warnings

from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

# ignore warnings
warnings.filterwarnings('ignore')
# Load digits dataset
iris = datasets.load_iris()
# # Create feature matrix
X = iris.data
# Create target vector
y = iris.target
# test size
test_size = 0.33
# generate the same set of random numbers
seed = 7
# cross-validation settings
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# Model instance
model = LogisticRegression()
# Evaluate model performance
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print('Accuracy -val set: %.2f%% (%.2f)' % (results.mean() * 100, results.std()))

# split data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)
# fit model
model.fit(X_train, y_train)
# accuracy on test set
result = model.score(X_test, y_test)
print("Accuracy - test set: %.2f%%" % (result * 100.0))
