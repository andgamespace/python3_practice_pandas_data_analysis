import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

df = pd.read_csv('practice_python_files/datasets/breast+cancer+wisconsin+original/breast-cancer-wisconsin.data')

df.columns = df.columns.str.strip()  # Strip leading/trailing spaces from column names
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)

# Print column names to debug
print(df.columns)

x = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1, -1)

prediction = clf.predict(example_measures)
print(prediction)