import pandas as pd

df = pd.read_csv('titanic.csv')

df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

inputs = df.drop('Survived', axis='columns')
target = df['Survived']

inputs['Sex'] = inputs['Sex'].map({'male': 1, 'female': 2})

inputs['Age'] = inputs['Age'].fillna(inputs['Age'].mean())

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
inputs_train, inputs_test, target_train, target_test = train_test_split(inputs, target, test_size=0.25, random_state=0)

from sklearn import tree

clf = tree.DecisionTreeClassifier()

# Train the model
clf.fit(inputs_train, target_train)

# Get the accuracy of the model
accuracy = clf.score(inputs_test, target_test)
print(f"Accuracy: {accuracy*100:.2f}%")

# Print confusion matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(target_test, clf.predict(inputs_test))

# Plot the confusion matrix using seaborn
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
sns.heatmap(matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

