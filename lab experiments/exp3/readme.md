
# Decision Tree using ID3 Algorithm
# Dataset, Text Output, Tree Visualization and Classification

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Step 1: Create dataset
data = {
    'Outlook': [
        'Sunny', 'Sunny', 'Overcast', 'Rain',
        'Rain', 'Rain', 'Overcast', 'Sunny',
        'Sunny', 'Rain', 'Sunny', 'Overcast',
        'Overcast', 'Rain'
    ],
    'Temperature': [
        'Hot', 'Hot', 'Hot', 'Mild',
        'Cool', 'Cool', 'Cool', 'Mild',
        'Cool', 'Mild', 'Mild', 'Mild',
        'Hot', 'Mild'
    ],
    'Humidity': [
        'High', 'High', 'High', 'High',
        'Normal', 'Normal', 'Normal', 'High',
        'Normal', 'Normal', 'Normal', 'High',
        'Normal', 'High'
    ],
    'Wind': [
        'Weak', 'Strong', 'Weak', 'Weak',
        'Weak', 'Strong', 'Strong', 'Weak',
        'Weak', 'Weak', 'Strong', 'Strong',
        'Weak', 'Strong'
    ],
    'Play': [
        'No', 'No', 'Yes', 'Yes',
        'Yes', 'No', 'Yes', 'No',
        'Yes', 'Yes', 'Yes', 'Yes',
        'Yes', 'No'
    ]
}

df = pd.DataFrame(data)

print("TRAINING DATASET:")
print(df.to_string(index=False))

# Step 2: Encode categorical attributes
encoded_df = df.copy()
encoders = {}

for column in encoded_df.columns:
    encoders[column] = LabelEncoder()
    encoded_df[column] = encoders[column].fit_transform(
        encoded_df[column]
    )

X = encoded_df.drop('Play', axis=1)
y = encoded_df['Play']

# Step 3: Build decision tree using entropy (ID3 criterion)
model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)
model.fit(X, y)

# Step 4: Display training accuracy
predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)

print("\nDECISION TREE CLASSIFICATION")
print("Training Accuracy:", round(accuracy * 100, 2), "%")

# Step 5: Visualize the decision tree
plt.figure(figsize=(16, 9))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=encoders['Play'].classes_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree using ID3 (Entropy)")
plt.show()

# Step 6: Classify a new sample
new_sample = {
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Wind': 'Strong'
}

new_df = pd.DataFrame([new_sample])

# Encode using the training data's encoders
for column in new_df.columns:
    new_df[column] = encoders[column].transform(new_df[column])

result = model.predict(new_df)[0]
predicted_class = encoders['Play'].inverse_transform([result])[0]

print("\nNEW SAMPLE:")
print(new_sample)
print("Predicted Class:", predicted_class)

# Step 7: Graph of class distribution
class_counts = df['Play'].value_counts()

plt.figure(figsize=(6, 4))
plt.bar(class_counts.index, class_counts.values)
plt.title("Distribution of Play Classes")
plt.xlabel("Play")
plt.ylabel("Number of Samples")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

print("\nResult: ID3 decision tree built and new sample classified.")
