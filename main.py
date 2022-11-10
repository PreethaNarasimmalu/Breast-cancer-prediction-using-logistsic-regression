#breast cancer prediction
#logistic regression
import numpy as np
import sklearn.datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

breast_cancer = sklearn.datasets.load_breast_cancer()
print(breast_cancer)

X=breast_cancer.data
Y=breast_cancer.target
print(X,Y)

data = pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
data['class']=breast_cancer.target
data.describe() #to get the statistical value of data
data.groupby('class').mean()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1) #we need only 10% of data as testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)

classifier=LogisticRegression()
classifier.fit(X_train,Y_train)

prediction_on_training_data = classifier.predict(X_train)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)
prediction_on_test_data = classifier.predict(X_test)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data)

print("Accuracy on training data : ",accuracy_on_training_data)
print("Accuracy on test data: ",accuracy_on_test_data)

#input data downloaded from kaggle
input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)
input_data_as_numpy_array = np.asarray(input_data) #asarray coverts this tuple to array
print(input_data_as_numpy_array)

input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)#converting it to a 3 d arrays
print(input_data_reshaped)

#prediction
prediction = classifier.predict(input_data_reshaped) #returns a list with element 0 or 1 (0-malignant,1-benign)
print(prediction)

if (prediction[0]==0):
  print("The breast cancer is malignant")
else:
  print("The breast cancer is benign")
