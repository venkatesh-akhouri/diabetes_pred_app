import numpy as np
import pandas as pd
import pickle

df=pd.read_csv("kaggle_diabetes.csv")
df=df.rename(columns={"DiabetesPedigreeFunction":"DPF","BloodPressure":"BP"})

df[["Glucose","BP","SkinThickness","Insulin","BMI"]]=df[["Glucose","BP","SkinThickness","Insulin","BMI"]].replace(0,np.NaN)

df["Glucose"].fillna(df["Glucose"].mean(),inplace=True)
df["BP"].fillna(df["BP"].mean(),inplace=True)
df["SkinThickness"].fillna(df["SkinThickness"].median(),inplace=True)
df["Insulin"].fillna(df["Insulin"].median(),inplace=True)
df["BMI"].fillna(df["BMI"].median(),inplace=True)

from sklearn.model_selection import train_test_split
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)





from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=49)
classifier.fit(X_train,Y_train)

pickle.dump(classifier,open("model.pkl","wb"))
