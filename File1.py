1) Logic gates:

def and_gate(x, y):
    return x and y

def or_gate(x, y):
    return x or y

def not_gate(x):
    return not x

def nand_gate(x, y):
    return not and_gate(x, y)

def nor_gate(x, y):
    return not or_gate(x, y)

def xor_gate(x, y):
    return x != y

def xnor_gate(x, y):
    return x == y

def equivalence(x, y):
    return x == y

def implication(x, y):
    return not x or y

def main():
    inputs = [(False, False), (False, True), (True, False), (True, True)]

    gates = {
        'AND': and_gate, 'OR': or_gate,
        'NOT (x,y)': lambda x, y: (not_gate(x), not_gate(y)),
        'NAND': nand_gate, 'NOR': nor_gate,
        'XOR': xor_gate, 'XNOR': xnor_gate,
        'EQUIVALENCE': equivalence, 'IMPLICATION': implication
    }

    for gate_name, gate_function in gates.items():
        print(f"\n{gate_name} Gate:")
        print("x   y   | output")
        print("-" * 20)

        for x, y in inputs:
            result = gate_function(x, y)
            if isinstance(result, tuple):
                print(f"{str(x):<5} {str(y):<5} | {str(result[0])}, {str(result[1])}")
            else:
                print(f"{str(x):<5} {str(y):<5} | {str(result)}")

if __name__ == "__main__":
    main()



2) recursion and recurrence relations: 

import matplotlib.pyplot as plt

def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

def generate_fibonacci_sequence(num_terms):
    fibonacci_numbers = []
    for i in range(num_terms):
        fibonacci_numbers.append(fibonacci(i))
    return fibonacci_numbers

def plot_fibonacci_sequence(fibonacci_numbers):
    plt.plot(fibonacci_numbers, marker='o', linestyle='-', color='b')
    plt.title("Fibonacci Sequence")
    plt.xlabel("Index (n)")
    plt.ylabel("Fibonacci Number F(n)")
    plt.grid(True)
    plt.show()

num_terms = 30
fibonacci_numbers = generate_fibonacci_sequence(num_terms)
print(fibonacci_numbers)
plot_fibonacci_sequence(fibonacci_numbers)


3) Solving Linear Programming Problem (LPP) Using Graphical Method: 

import numpy as np
import matplotlib.pyplot as plt

# Input objective and constraints
c1 = float(input("Enter c1: "))
c2 = float(input("Enter c2: "))
a1, b1, c1_con = map(float, input("Enter a1 b1 c1 (constraint 1): ").split())
a2, b2, c2_con = map(float, input("Enter a2 b2 c2 (constraint 2): ").split())

# Define constraint lines
def line1(x):
    return (c1_con - a1*x) / b1 if b1 != 0 else float('inf')

def line2(x):
    return (c2_con - a2*x) / b2 if b2 != 0 else float('inf')

# Find intersection points with axes
points = [(0, 0)]
if b1 != 0:
    points.append((0, c1_con/b1))
if a1 != 0:
    points.append((c1_con/a1, 0))
if b2 != 0:
    points.append((0, c2_con/b2))
if a2 != 0:
    points.append((c2_con/a2, 0))

# Check valid points (satisfying both constraints)
valid = [p for p in points if a1*p[0] + b1*p[1] <= c1_con and a2*p[0] + b2*p[1] <= c2_con]

# Calculate Z values for valid points
z = [c1*x + c2*y for x, y in valid]
opt = valid[np.argmax(z)] if z else None

# Plotting
x = np.linspace(0, 10, 400)
y1 = [line1(i) for i in x]
y2 = [line2(i) for i in x]

plt.plot(x, y1, label="Constraint 1")
plt.plot(x, y2, label="Constraint 2")

plt.fill_between(x, 0, np.minimum(y1, y2), where=(np.minimum(y1, y2) >= 0), color='lightgreen', alpha=0.5, label="Feasible Region")

for p in valid:
    plt.plot(*p, 'ko')
if opt:
    plt.plot(*opt, 'ro', label="Optimal")

plt.legend()
plt.grid()
plt.title("LPP Graph")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(bottom=0)
plt.show()

# Output results
if opt:
    print("Optimal Point:", opt)
    print("Max Z =", c1*opt[0] + c2*opt[1])
else:
    print("No feasible solution.")

OUTPUT:
Enter the coefficient for x in the objective function: 3 
Enter the coefficient for y in the objective function: 2 
Enter constraints of the form ax + by ≤ c
Enter a1, b1, c for constraint 1: 1 1 4
Enter a2, b2, c for constraint 2: 0 1 4


4) Solving Linear Programming Problem (LPP) Using Simplex Method: 

import numpy as np
from scipy.optimize import linprog

# Get coefficients for the objective function
c = list(map(float, input("Enter objective function coefficients (separated by space): ").split()))
c = [-x for x in c]  # Convert to minimization

# Get number of constraints
n = int(input("Enter number of constraints: "))
A = []
b = []

print("\nEnter each constraint as: a1 a2 ... an <= b")
for i in range(n):
    parts = list(map(float, input(f"Constraint {i+1}: ").split()))
    A.append(parts[:-1])
    b.append(parts[-1])

# Variable bounds: all variables >= 0
bounds = [(0, None)] * len(c)

# Solve LP problem
res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

# Output result
if res.success:
    print("\nOptimal solution found:")
    for i, val in enumerate(res.x):
        print(f"x{i+1} = {val:.4f}")
    print(f"Maximum Z = {-res.fun:.4f}")
else:
    print("No optimal solution found.")


Output:
Enter obj funct coeffic 3 4 
Enter number of constraints: 2 
Constraint 1: 2 1 10
Constraint 2: 1 2 8 
Optimal solution found: 
x1 = 4.0000
x2 = 2.0000
Max Z = 20.0000


5) Find optimum solution for balanced and Unbalanced Transportation 

import numpy as np

def northwest_corner(supply, demand):
    alloc = np.zeros((len(supply), len(demand)))
    i = j = 0
    while i < len(supply) and j < len(demand):
        val = min(supply[i], demand[j])
        alloc[i][j] = val
        supply[i] -= val
        demand[j] -= val
        if supply[i] == 0:
            i += 1
        if demand[j] == 0:
            j += 1
    return alloc

def least_cost(cost, supply, demand):
    alloc = np.zeros_like(cost)
    cells = sorted(((i, j) for i in range(len(supply)) for j in range(len(demand))),
                   key=lambda x: cost[x])
    for i, j in cells:
        if supply[i] and demand[j]:
            val = min(supply[i], demand[j])
            alloc[i][j] = val
            supply[i] -= val
            demand[j] -= val
    return alloc

def total_cost(cost, alloc):
    return np.sum(cost * alloc)

def transport():
    m = int(input("Enter number of sources: "))
    n = int(input("Enter number of destinations: "))
    
    supply = list(map(int, input("Enter supply: ").split()))
    demand = list(map(int, input("Enter demand: ").split()))
    
    print("Enter cost matrix (each row):")
    cost = [list(map(int, input().split())) for _ in range(m)]
    cost = np.array(cost)
    
    print("1. Northwest Corner\n2. Least Cost Method")
    ch = int(input("Choose method (1/2): "))
    
    if ch == 1:
        alloc = northwest_corner(supply[:], demand[:])
    elif ch == 2:
        alloc = least_cost(cost, supply[:], demand[:])
    else:
        print("Invalid choice.")
        return
    
    print("\nAllocation:\n", alloc)
    print("Total Cost:", total_cost(cost, alloc))

transport()

OUTPUT:
Enter number of sources: 2 
Enter number of destinations: 3 
Enter supply: 20 30
Enter demand: 10 25 15 
Enter cost matrix:
8 6 10
9 7 4
1.Northwest Corner
2.Least Cost Method Choose method (1/2): 2 Allocation:
[[ 0 20 0]
[10 5 15]]
Total Cost: 305


6) Solving Linear Programming Problems Using the Two Phase Simplex Method

import numpy as np
from scipy.optimize import linprog

def solve_lp(c, A, b, A_eq=None, b_eq=None, bnds=None):
    print("\nSolving with Two-Phase Simplex Method...\n")
    r = linprog(c, A_ub=A, b_ub=b,
                A_eq=A_eq, b_eq=b_eq,
                bounds=bnds, method='highs')
    if r.success:
        print("Optimal solution found:")
        print("Variables:", r.x)
        print("Objective value:", -r.fun)
    else:
        print("The problem is infeasible or unbounded.")

n_var = int(input("Enter number of variables: "))
n_con = int(input("Enter number of constraints: "))

c = list(map(float, input(f"Enter {n_var} coefficients of the objective function (space-separated): ").split()))
c = [-v for v in c]

A = []
for i in range(n_con):
    row = list(map(float, input(f"Enter {n_var} coefficients for constraint {i+1} (space-separated): ").split()))
    A.append(row)

b = list(map(float, input(f"Enter {n_con} right-hand side values of constraints (space-separated): ").split()))

bnds = [(0, None)] * n_var

solve_lp(np.array(c), np.array(A), np.array(b), bnds=bnds)

OUTPUT:
Enter number of variables: 2
Enter number of constraints: 2
Enter 2 coefficients of the objective function (space-separated): 4 5
Enter 2 coefficients for constraint 1 (space-separated): 2 3
Enter 2 coefficients for constraint 2 (space-separated): -3 -1
Enter 2 right-hand side values of constraints (space-separated): 6 3

Solving with Two-Phase Simplex Method...

Optimal solution found:
Variables: [3. 0.]
Objective value: 12.0


7) Solving Assignment Problems Using the Hungarian Algorithm

import numpy as np
from scipy.optimize import linear_sum_assignment

n = int(input("Enter number of agents/tasks: "))
print(f"Enter the cost matrix ({n} x {n}):")

cost_matrix = [list(map(int, input().split())) for _ in range(n)]
cost_matrix = np.array(cost_matrix)

row_ind, col_ind = linear_sum_assignment(cost_matrix)
total_cost = cost_matrix[row_ind, col_ind].sum()

print("\nOptimal Assignment:")
for i, j in zip(row_ind, col_ind):
    print(f"Agent {i+1} assigned to Task {j+1} | Cost = {cost_matrix[i][j]}")

print(f"Total Minimum Cost: {total_cost}")

OUTPUT:
Enter number of agents/tasks: 3
Enter the cost matrix (3 x 3):
4 2 8
4 3 7
3 1 6

Optimal Assignment:
Agent 1 assigned to Task 2 | Cost = 2
Agent 2 assigned to Task 1 | Cost = 4
Agent 3 assigned to Task 3 | Cost = 6
Total Minimum Cost: 12


8) Determining cpm and pert

import networkx as nx
import matplotlib.pyplot as plt

# Ensure matplotlib plots are displayed inline in Jupyter Notebook
%matplotlib inline

def expected(o, m, p): 
    return (o + 4 * m + p) / 6

def compute(acts):
    G, dur = nx.DiGraph(), {}
    for t, o, m, p, pre in acts:
        d = expected(o, m, p); dur[t] = d; G.add_node(t)
        for pr in pre: 
            G.add_edge(pr.strip(), t) if pr.strip() else None
    es, lf = {}, {}
    for n in nx.topological_sort(G):
        es[n] = max([es.get(p, 0) + dur[p] for p in G.predecessors(n)], default=0)
    total = max([es[n] + dur[n] for n in G])
    for n in G: 
        lf[n] = total
    for n in reversed(list(nx.topological_sort(G))):
        for s in G.successors(n): 
            lf[n] = min(lf[n], lf[s] - dur[n])
    slack = {n: lf[n] - es[n] for n in G}
    cp = [n for n in G if slack[n] == 0]
    pert_total = sum(dur.values())
    return G, cp, total, dur, slack, pert_total

def draw(G, cp):
    pos = nx.spring_layout(G)
    # Initialize the plot
    plt.figure(figsize=(8, 6))
    color = ['red' if node in cp else 'lightblue' for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=color, node_size=1500, arrows=True)
    plt.title("Critical Path")
    plt.show()

# Input from user
n = int(input("Number of tasks: "))
acts = []
for _ in range(n):
    t = input("Task name: ")
    o = float(input("Optimistic time: "))
    m = float(input("Most likely time: "))
    p = float(input("Pessimistic time: "))
    pre = input("Predecessors (comma-separated, blank if none): ").split(",") if input("Any predecessors? (y/n): ") == 'y' else []
    acts.append((t, o, m, p, pre))

# Compute and display results
G, cp, cpm_dur, durations, slack, pert_dur = compute(acts)
print("\n--- CPM Results ---")
print("Critical Path:", ' → '.join(cp))
print("Project Duration (CPM):", round(cpm_dur, 2))
print("Slack Times:", slack)
print("\n--- PERT Results ---")
print("Expected Project Duration (PERT):", round(pert_dur, 2))
print("Activity Durations:", durations)

# Draw the network graph
draw(G, cp)

OUTPUT:

Number of tasks: 5

Task name: A
Optimistic time: 2
Most likely time: 4
Pessimistic time: 6
Any predecessors? (y/n): n

Task name: B
Optimistic time: 3
Most likely time: 5
Pessimistic time: 9
Any predecessors? (y/n): y
Predecessors (comma-separated, or blank if none): A

Task name: C
Optimistic time: 1
Most likely time: 2
Pessimistic time: 3
Any predecessors? (y/n): y
Predecessors (comma-separated, or blank if none): A

Task name: D
Optimistic time: 2
Most likely time: 3
Pessimistic time: 8
Any predecessors? (y/n): y
Predecessors (comma-separated, or blank if none): B,C

Task name: E
Optimistic time: 3
Most likely time: 6
Pessimistic time: 9
Any predecessors? (y/n): y
Predecessors (comma-separated, or blank if none): D


9.Linear Model 

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('house-prices.csv')

# Handling Missing Values (if any)
data = data.dropna()  # or you can use data.fillna(method='ffill') if required

# If there are categorical columns, convert them (example)
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].astype('category').cat.codes

# Correlation and Correlation Matrix Visualization
corr_matrix = data.corr()
print("Correlation Matrix:\n", corr_matrix)

# Plot heatmap
plt.figure(figsize=(6,4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Prepare feature and target
X = data['area'].values.reshape(-1, 1)
y = data['price'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-Squared Score: {r_squared:.2f}")

# One Visualization: Scatter Plot with Regression Line
plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Testing Data')
plt.plot(X_test, y_pred, color='green', label='Regression Line')
plt.xlabel('House Size (sq ft)')
plt.ylabel('House Price')
plt.title('House Size vs Price')
plt.legend()
plt.show()

# Residual Plot
plt.figure(figsize=(6,4))
plt.scatter(y_pred, y_test - y_pred, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

# Predicting a Single Value (example: area = 3000 sq ft)
new_area = np.array([[3000]])
predicted_price = model.predict(new_area)
print(f"Predicted price for house area {new_area[0][0]} sq ft: {predicted_price[0]:.2f}")


EXTRA:
# Drop missing values only in 'area' column
data = data.dropna(subset=['area'])

# Convert only 'city' column from categorical to numerical
data['city'] = data['city'].astype('category').cat.codes


[OR]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/MC Lab Final/EV_Performance_Dataset.csv")

# Prepare the features and target
X = df[["Battery_Capacity_kWh"]]
y = df["EV_Range_km"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Plot the results
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_test.squeeze(), y=y_test, label="Actual")
sns.lineplot(x=X_test.squeeze(), y=y_pred, color="red", label="Predicted")
plt.xlabel("Battery Capacity (kWh)")
plt.ylabel("EV Range (km)")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

# Save the model
model_dir = '/content/drive/MyDrive/MC Lab Final/Model Save'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'ev_range_model.pkl')
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

# Load the saved model
loaded_model = joblib.load(model_path)

# Predict the EV range for a given battery capacity
battery_input = np.array([[75]])  # Example input: 75 kWh
predicted_range = loaded_model.predict(battery_input)
print(f"Predicted EV Range for 75 kWh: {predicted_range[0]:.2f} km")


10) Multi model:

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('Student_Performance.csv')

# Handling Missing Values (drop missing rows in specific important columns)
data = data.dropna(subset=['Hours Studied', 'Previous Scores'])  # Example, adjust based on your columns

# Convert Categorical Columns to Numerical (if any)
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].astype('category').cat.codes

# Show Correlation Matrix
corr_matrix = data.corr()
print("\nCorrelation Matrix:\n", corr_matrix)

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Prepare X and y
X = data.drop('Performance Index', axis=1)  # All columns except target
y = data['Performance Index']               # Target column

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Multilinear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on Test Set
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Data Visualization: Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual Performance')
plt.ylabel('Predicted Performance')
plt.title('Actual vs Predicted')
plt.show()

new_data = np.array([[6, 80, 1, 7, 10]])  

# Now predict
predicted_performance = model.predict(new_data)
print(f"\nPredicted Performance Index for the new student: {predicted_performance[0]:.2f}")

EXTRA :

# Drop rows where 'Gender' is missing
data = data.dropna(subset=['Gender'])

# Convert 'Gender' column to numerical
data['Gender'] = data['Gender'].astype('category').cat.codes


# Example: Predicting new student's performance
new_data = np.array([[6, 80, 1, 7, 10]])  # 6 hours study, 80 previous score, activity=1, 7 sleep hours, 10 practice papers
predicted_value = model.predict(new_data)

print(f"Predicted Performance Index: {predicted_value[0]:.2f}")



[OR]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/MC Lab Final/EV_Performance_Dataset.csv")

# Data Preprocessing
num_features = ["Battery_Capacity_kWh", "Vehicle_Weight_kg", "Tire_Pressure_psi", "Ambient_Temperature_C", "Drag_Coefficient", "Regen_Braking_Efficiency_%"]
cat_features = ["EV_Brand", "Charging_Type", "Road_Type", "Battery_Manufacturer", "Drive_Mode"]

# Define transformations for numeric and categorical features
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine transformers into a preprocessor pipeline
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Split the data into features (X) and target (y)
X = df[num_features + cat_features]
y = df["EV_Range_km"]

# Train, validation, and test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# Build the model pipeline with preprocessing and linear regression
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the model
model.fit(X_train, y_train)

# Cross-validation
cross_val = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-validation R2 Scores: {cross_val}")
print(f"Mean CV R2 Score: {cross_val.mean():.2f}")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual EV Range (km)")
plt.ylabel("Predicted EV Range (km)")
plt.axline((0, 0), slope=1, color='red', linestyle='dashed')
plt.show()

# Save the model
model_dir = '/content/drive/MyDrive/MC Lab Final/Model Save'
os.makedirs(model_dir, exist_ok=True)
mlr_model_path = os.path.join(model_dir, 'ev_mlr_model.pkl')
joblib.dump(model, mlr_model_path)
print(f"MLR model saved to: {mlr_model_path}")

# Load the saved model
loaded_model = joblib.load(mlr_model_path)

# Predict using the loaded model for new input
new_input = pd.DataFrame([[75, 1800, 36, 25, 0.29, 85, "Tesla", "Fast", "Highway", "LG Chem", "Sport"]],
                         columns=X.columns)
predicted_range = loaded_model.predict(new_input)
print(f"Predicted EV Range: {predicted_range[0]:.2f} km") 

#pip install networkx matplotlib
