#Rnadom forest classifier for dementia diagnosis prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\Users\Ruthy\Desktop\dementia data set numerical only.csv")
print(df)

X = df.drop('Dementia', axis=1) #features
y = df['Dementia'] #target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

rf = RandomForestClassifier(random_state=13)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

