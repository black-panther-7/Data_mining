import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn

# Load the dataset
data_set = pd.read_csv("cancer.csv")

# Data preprocessing to the dataset
data_set = data_set.dropna(axis=1)
data_set = data_set.drop('id', axis=1)

# Assigning the data to X(input features) and y(target variable) values
X = data_set.drop(columns='diagnosis')
y = data_set["diagnosis"]

# X is the feature matrix and y is the target vector
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=4)

# create a linear SVM
svm_linear = svm.SVC(kernel='poly', degree=6, gamma='auto', max_iter=3)
#svm_linear = svm.SVC(kernel='poly', max_iter=1, gamma='auto')


# train the model
svm_linear.fit(X_train, y_train)

# predict using the trained model
predictions = svm_linear.predict(X_test)

# finding the accuracy
acc_score = accuracy_score(y_test, predictions)
print(acc_score)

# evaluate the model
print(classification_report(y_test, predictions))

# confusion matrix for the model
confusion_matrix_cancer = confusion_matrix(y_test, predictions)

# Plotting heat map for confusion matrix
plt.figure(figsize=(7, 7))
sn.heatmap(confusion_matrix_cancer, annot=True, fmt='d')
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Y', fontsize=13)
plt.ylabel('Actual Y', fontsize=13)
plt.savefig("confusion_matrix.png")
plt.show()
