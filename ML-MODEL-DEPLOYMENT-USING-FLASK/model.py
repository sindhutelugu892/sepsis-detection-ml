# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
# import pickle
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.utils import resample

# # Load the csv file
# df = pd.read_csv("data1.csv")

# print(df.head())

# majority_class = df[df.SepsisLabel == 0]
# minority_class = df[df.SepsisLabel == 1]
# print('number of sepsis label 1 is {}'.format(len(minority_class)))
# print('while number of sepsis label 0 is {}'.format(len(majority_class)))
# data_minority_upsampled=resample(minority_class, replace=True, n_samples=30568, random_state=123)
# data_upsampled=pd.concat([majority_class,data_minority_upsampled])

# # majority_class_subset = majority_class.sample(n=2*len(minority_class))
# # df = pd.concat([majority_class_subset, minority_class])

# print('number of sepsis label 1 is {}'.format(len(data_minority_upsampled)))
# print('while number of sepsis label 0 is {}'.format(len(majority_class)))

# # Select independent and dependent variable
# # X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
# # y = df["Class"]

# X = df[["HR", "O2Sat", "Temp", "MAP", "Resp","BUN","Chloride","Creatinine","Glucose","Hct","Hgb","WBC","Platelets","Age","Unit1","Unit2","HospAdmTime","ICULOS"]]
# y = df["SepsisLabel"]

# # X = df[["HR", "O2Sat", "Temp", "MAP", "Resp","Age", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]]
# # y = df["SepsisLabel"]

# # Split the dataset into train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# # Feature scaling
# # sc = StandardScaler()
# # X_train = sc.fit_transform(X_train)
# # X_test= sc.transform(X_test)

# # Instantiate the model
# classifier = RandomForestClassifier(n_estimators=300, random_state=0)

# # Fit the model
# classifier.fit(X_train, y_train)

# pred_rfc = classifier.predict(X_test)
# acc_rfc = accuracy_score(y_test, pred_rfc) # evaluating accuracy score
# print('accuracy score of RandomForest Classifier is:', acc_rfc * 100)
# # Make pickle file of our model
# pickle.dump(classifier, open("model.pkl", "wb"))


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB

# Load the csv file
df = pd.read_csv("data1.csv")

print(df.head())

majority_class = df[df.SepsisLabel == 0]
minority_class = df[df.SepsisLabel == 1]
print('number of sepsis label 1 is {}'.format(len(minority_class)))
print('while number of sepsis label 0 is {}'.format(len(majority_class)))
data_minority_upsampled=resample(minority_class, replace=True, n_samples=30568, random_state=123)
data_upsampled=pd.concat([majority_class,data_minority_upsampled])

print('number of sepsis label 1 is {}'.format(len(data_minority_upsampled)))
print('while number of sepsis label 0 is {}'.format(len(majority_class)))

X = df[["HR", "O2Sat", "Temp", "MAP", "Resp","BUN","Chloride","Creatinine","Glucose","Hct","Hgb","WBC","Platelets","Age","Unit1","Unit2","HospAdmTime","ICULOS"]]
y = df["SepsisLabel"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# KNC = KNeighborsClassifier() 
# NB = GaussianNB() 
RFC = RandomForestClassifier() #intialising Random Forest Classifier

model_RandomForestClassifier = RFC.fit(X_train, y_train) # fitting Training Set
pred_rfc = model_RandomForestClassifier.predict(X_test)
acc_rfc = accuracy_score(y_test, pred_rfc) # evaluating accuracy score
print('accuracy score of RandomForest Classifier is:', acc_rfc * 100)
pickle.dump(pred_rfc, open("model.pkl", "wb"))

# model_kNeighborsClassifier = KNC.fit(X_train, y_train) # fitting Training Set
# pred_knc = model_kNeighborsClassifier.predict(X_test) # Predicting on test dataset
# acc_knc = accuracy_score(y_test, pred_knc) # evaluating accuracy score
# print('accuracy score of KNeighbors Classifier is:', acc_knc * 100)

# model_NaiveBayes = NB.fit(X_train, y_train)
# pred_nb = model_NaiveBayes.predict(X_test)
# acc_nb = accuracy_score(y_test, pred_nb)
# print('Accuracy of Naive Bayes Classifier:', acc_nb * 100)

# from mlxtend.classifier import StackingClassifier
# lr = LogisticRegression() # defining meta-classifier
# clf_stack = StackingClassifier(classifiers =[KNC,NB,RFC], meta_classifier = lr, use_probas = True, use_features_in_secondary = True)
# model_stack = clf_stack.fit(X_train, y_train) # training of stacked model
# pred_stack = model_stack.predict(X_test)	 # predictions on test data using stacked model
# acc_stack = accuracy_score(y_test, pred_stack) # evaluating accuracy
# print('accuracy score of Stacked model:', acc_stack * 100)
