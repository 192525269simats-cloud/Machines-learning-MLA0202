
import math
import turtle

# Training dataset: (X, Y, Class)
data = [
    (1, 2, "A"),
    (2, 3, "A"),
    (3, 2, "A"),
    (3, 4, "A"),
    (6, 7, "B"),
    (7, 8, "B"),
    (8, 6, "B"),
    (7, 5, "B")
]

# Test point
test_point = (4, 5)
k = 3

# Calculate Euclidean distance
distances = []
for x, y, label in data:
    d = math.sqrt(
        (x - test_point[0]) ** 2 +
        (y - test_point[1]) ** 2
    )
    distances.append((d, label))

# Sort by distance
distances.sort(key=lambda item: item[0])

# Select K nearest neighbours
neighbors = distances[:k]

# Majority voting
count_A = sum(1 for d, label in neighbors if label == "A")
count_B = sum(1 for d, label in neighbors if label == "B")

predicted_class = "A" if count_A > count_B else "B"

print("K-Nearest Neighbours Algorithm")
print("Test Point:", test_point)
print("K value:", k)
print("Nearest Neighbours:", neighbors)
print("Predicted Class:", predicted_class)

# Draw graph using turtle
screen = turtle.Screen()
screen.title("K-NN Classification Graph")
screen.setup(700, 650)
screen.setworldcoordinates(-1, -1, 11, 11)
screen.bgcolor("white")

pen = turtle.Turtle()
pen.speed(0)
pen.hideturtle()

# Draw coordinate axes
pen.penup()
pen.goto(0, 0)
pen.pendown()
pen.pensize(2)
pen.color("black")
pen.goto(10, 0)
pen.penup()
pen.goto(0, 0)
pen.pendown()
pen.goto(0, 10)

# Draw axis labels
pen.penup()
pen.goto(10, -0.5)
pen.write("X", align="center", font=("Arial", 12, "bold"))
pen.goto(-0.5, 10)
pen.write("Y", align="center", font=("Arial", 12, "bold"))

# Plot training points
for x, y, label in data:
    pen.penup()
    pen.goto(x, y)
    pen.dot(14, "blue" if label == "A" else "green")
    pen.goto(x + 0.15, y + 0.2)
    pen.write(label, font=("Arial", 10, "normal"))

# Plot test point
pen.penup()
pen.goto(test_point)
pen.dot(18, "red")
pen.goto(test_point[0] + 0.2, test_point[1] + 0.2)
pen.write("Test", font=("Arial", 10, "bold"))

# Highlight K nearest neighbours
for d, label in neighbors:
    for x, y, actual_label in data:
        if actual_label == label:
            distance = math.sqrt(
                (x - test_point[0]) ** 2 +
                (y - test_point[1]) ** 2
            )
            if abs(distance - d) < 0.000001:
                pen.penup()
                pen.goto(x, y)
                pen.pendown()
                pen.color("orange")
                pen.pensize(2)
                pen.goto(test_point)
                pen.color("black")
                break

# Legend
pen.penup()
pen.goto(0, -0.7)
pen.write(
    "Blue = Class A | Green = Class B | Red = Test Point",
    font=("Arial", 10, "normal")
)

turtle.done()
