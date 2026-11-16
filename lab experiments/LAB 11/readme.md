
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Dataset
data = {
    'Income': [25000, 40000, 50000, 60000, 20000, 70000, 80000, 30000],
    'Debt': [20000, 15000, 10000, 5000, 25000, 8000, 4000, 18000],
    'Payment': [1, 1, 1, 0, 0, 1, 1, 0],
    'Credit': ['Poor', 'Good', 'Good', 'Poor',
               'Poor', 'Good', 'Good', 'Poor']
}

df = pd.DataFrame(data)

# Training
X = df[['Income', 'Debt', 'Payment']]
y = df['Credit']

model = DecisionTreeClassifier()
model.fit(X, y)

# Prediction
pred = model.predict(X)

print("Credit Score Classification:")
print(pred)

# Graph
df['Predicted'] = pred
df['Predicted'].value_counts().plot(kind='bar')

plt.title("Credit Score Classification")
plt.xlabel("Credit Score")
plt.ylabel("Number of People")
plt.show()
