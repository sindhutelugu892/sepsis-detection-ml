import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

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

RFC = RandomForestClassifier() #intialising Random Forest Classifier

model_RandomForestClassifier = RFC.fit(X_train, y_train) # fitting Training Set
pred_rfc = model_RandomForestClassifier.predict(X_test)
acc_rfc = accuracy_score(y_test, pred_rfc) # evaluating accuracy score
print('accuracy score of RandomForest Classifier is:', acc_rfc * 100)

pickle.dump(pred_rfc, open("model.pkl", "wb"))