import os
import pickle
import pandas as pd
import numpy as np
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from itertools import product
from sklearn.linear_model import LogisticRegression
from collections import defaultdict

all_files = glob.glob("archive/*.csv")
dfs = []
for filename in all_files:
    df = pd.read_csv(filename)
    df.fillna('unknow', inplace=True)
    df.rename(columns= {'old_column_name': 'new_column_name'}, inplace=True)
    dfs.append(df)
    index = ['row1']

combined_df = pd.concat(dfs, ignore_index=True)
print('print combined:',combined_df)
#separar os dados 
X = combined_df['text']
y = combined_df['class']
print(combined_df['class'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(y_test)

#Vectorizar os dados
vectorized = CountVectorizer()
X_train_transformed = vectorized.fit_transform(X_train)
X_test_transformed = vectorized.transform(X_test)
print(X_train_transformed)

#criar o modelo multinomial
clf = MultinomialNB()
clf.fit(X_train_transformed, y_train)

#predições do conjunto de provas
y_pred = clf.predict(X_test_transformed)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)

#aplicar Greedy Search na base de treino e acurácia
gs = SVC()
best_alpha = 0
best_accuracy = 0

for alpha in [0.1, 0.5, 1.0, 2.0]:
    gs = MultinomialNB(alpha=alpha)
    gs.fit(X_train_transformed, y_train)
    y_pred = gs.predict(X_test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_alpha = alpha

print("Best alpha:", best_alpha)
print("Best accuracy:", best_accuracy) 

#calculando metricas com F1_Score(micro e macro)
f1_micro= f1_score(y_test, y_pred, average="micro") 
f1_macro = f1_score(y_test, y_pred, average="macro")
print("F1-score micro:", f1_micro)
print("F1-score macro:", f1_macro)

metricas = pd.DataFrame(columns=['Best_alpha','Best_accuracy' 'F1-score Micro', 'F1-score Macro'])

novos_resultados ={'Best_alpha': best_alpha, 'Best_accuracy': best_accuracy,'F1-score Micro': f1_score(y_test,y_pred, average="micro"), 'F1-score Macro': f1_score(y_test, y_pred, average="macro")}
index = ['row1']
df = pd.DataFrame(novos_resultados, index)


print(df)
df.to_csv("resultados_Sports.csv", index=False)
