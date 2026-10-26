
# Candidate-Elimination Algorithm with Graphs

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create training dataset and save as CSV
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'Temperature': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)
df.to_csv('training_data.csv', index=False)

# Step 2: Read training data from CSV
df = pd.read_csv('training_data.csv')
print("TRAINING DATA:")
print(df.to_string(index=False))

attributes = list(df.columns[:-1])
target = df.columns[-1]
num_attributes = len(attributes)

# Step 3: Initialize boundaries
S = ['0'] * num_attributes
G = [['?'] * num_attributes]

# Check if a hypothesis is more general than another
def more_general_or_equal(h1, h2):
    return all(
        a == '?' or a == b or b == '0'
        for a, b in zip(h1, h2)
    )

# Remove duplicate hypotheses
def remove_duplicates(hypotheses):
    result = []
    for h in hypotheses:
        if h not in result:
            result.append(h)
    return result

# Check whether a hypothesis covers an example
def covers(h, example):
    return all(
        h[i] == '?' or h[i] == example[i]
        for i in range(len(h))
    )

# Step 4: Candidate-Elimination
s_history = []
g_history = []

for index, row in df.iterrows():
    example = row.iloc[:-1].tolist()
    label = row.iloc[-1]

    print(f"\nExample {index + 1}: {example} -> {label}")

    if label == 'Yes':
        # Remove inconsistent hypotheses from G
        G = [g for g in G if covers(g, example)]

        # Generalize S minimally
        if S == ['0'] * num_attributes:
            S = example.copy()
        else:
            for i in range(num_attributes):
                if S[i] != example[i]:
                    S[i] = '?'

        # Remove S hypotheses that are not bounded by G
        G = [
            g for g in G
            if more_general_or_equal(g, S)
        ]

    else:
        # Remove hypotheses in S that cover negative example
        if S != ['0'] * num_attributes and covers(S, example):
            for i in range(num_attributes):
                if S[i] == '?':
                    S[i] = example[i]

        # Specialize G to exclude negative example
        new_G = []
        for g in G:
            if covers(g, example):
                for i in range(num_attributes):
                    if g[i] == '?':
                        if S[i] != '?' and S[i] != '0':
                            if S[i] != example[i]:
                                h = g.copy()
                                h[i] = S[i]
                                if more_general_or_equal(h, S):
                                    new_G.append(h)
            else:
                new_G.append(g)

        G = remove_duplicates(new_G)

    s_history.append(sum(v == '?' for v in S))
    g_history.append(len(G))

    print("Specific Boundary S:", S)
    print("General Boundary G:", G)

print("\nFINAL VERSION SPACE BOUNDARIES")
print("S =", S)
print("G =", G)

# Step 5: Graph 1 - Specific boundary generalization
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(s_history) + 1), s_history, marker='o')
plt.title("Candidate-Elimination: Specific Boundary")
plt.xlabel("Training Example")
plt.ylabel("Number of '?' in S")
plt.xticks(range(1, len(s_history) + 1))
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Graph 2 - Number of hypotheses in G
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(g_history) + 1), g_history, marker='o')
plt.title("Candidate-Elimination: General Boundary")
plt.xlabel("Training Example")
plt.ylabel("Number of Hypotheses in G")
plt.xticks(range(1, len(g_history) + 1))
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nResult: Candidate-Elimination algorithm executed.")
