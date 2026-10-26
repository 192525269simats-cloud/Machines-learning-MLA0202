
# FIND-S Algorithm with Graphical Representation

import matplotlib.pyplot as plt

# Training data
training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

attributes = [
    'Sky', 'Temperature', 'Humidity',
    'Wind', 'Water', 'Forecast'
]

# Initialize hypothesis
hypothesis = ['0'] * len(attributes)

# Store the number of generalized attributes at each step
steps = []
generalization = []
step = 0

print("FIND-S ALGORITHM")
print("-" * 40)
print("\nTraining Data:")

for row in training_data:
    print(row)

print("\nHypothesis Updates:")

for example in training_data:
    if example[-1] == 'Yes':
        step += 1

        for i in range(len(attributes)):
            if hypothesis[i] == '0':
                hypothesis[i] = example[i]
            elif hypothesis[i] != example[i]:
                hypothesis[i] = '?'

        # Count generalized attributes
        generalized = hypothesis.count('?')
        steps.append(step)
        generalization.append(generalized)

        print(f"Step {step}: {hypothesis}")

print("\nFinal Most Specific Hypothesis:")
print(hypothesis)

print("\nResult: FIND-S algorithm executed successfully.")

# Graph: number of generalized attributes
plt.figure(figsize=(8, 5))
plt.plot(steps, generalization, marker='o')
plt.title("FIND-S Algorithm - Hypothesis Generalization")
plt.xlabel("Positive Training Example")
plt.ylabel("Number of Generalized Attributes (?)")
plt.xticks(steps)
plt.grid(True)
plt.tight_layout()
plt.show()

# Graph: final hypothesis representation
plt.figure(figsize=(9, 5))
plt.bar(attributes, [
    0 if value == '0' else 1 if value == '?' else 2
    for value in hypothesis
])
plt.title("Final FIND-S Hypothesis")
plt.xlabel("Attributes")
plt.ylabel("Hypothesis Type")
plt.yticks([0, 1, 2], ['Specific (0)', 'General (?)', 'Fixed Value'])
plt.xticks(rotation=30)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
