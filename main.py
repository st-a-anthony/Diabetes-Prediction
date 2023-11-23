import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv(r'C:\Users\ASUSU\PycharmProjects\Diabetes Prediction\diabetes.csv') 

# printing the first 5 rows of the dataset
diabetes_dataset.head()

# number of rows and Columns in this dataset
diabetes_dataset.shape

# getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# print(X)
# print(Y)

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
# print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']
# print(X)
# print(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
# print(X.shape, X_train.shape, X_test.shape)
classifier = svm.SVC(kernel='linear')
#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)


def prediction(input_data):
  # input_data = (1,103,30,38,83,43.3,0.183,33)
  # changing the input_data to numpy array
  input_data_as_numpy_array = np.asarray(input_data)

  # reshape the array as we are predicting for one instance
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

  # standardize the input data
  std_data = scaler.transform(input_data_reshaped)
  # print(std_data)

  prediction = classifier.predict(std_data)
  # print(prediction)

  if (prediction[0] == 0):
    return 0
  else:
    return 1

# input_data=(6,148,72,35,0,33.6,0.627,50)
# input_data2=(1,85,66,29,0,26.6,0.351,31)
# prediction(input_data)
# prediction(input_data2)