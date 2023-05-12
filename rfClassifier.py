import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df_final = pd.read_csv('./data/HebrewSentences.csv')
X = df_final.drop(['sentence', 'result'], axis=1)
y = df_final['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022, stratify=y)

rf = RandomForestClassifier()
classifiers = rf

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
print(report)

df_confusion = confusion_matrix(y_test, y_pred)
pd.DataFrame(df_confusion).to_csv('confusion_matrix.csv')

group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                df_confusion.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     df_confusion.flatten()/np.sum(df_confusion)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
ax = sns.heatmap(df_confusion, annot=labels, fmt='', cmap='Blues')
ax.set_xlabel('True labels')
ax.set_ylabel('Predicted labels')
ax.xaxis.set_ticklabels(['Not Complex', 'Complex'])
ax.yaxis.set_ticklabels(['Not Complex', 'Complex'])
plt.show()
