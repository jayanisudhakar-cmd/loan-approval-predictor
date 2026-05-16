# app.py
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Features: [Income (k$), Credit Score, Loan Amount (k$)]
X = np.array([[50, 600, 10], [100, 750, 20], [30, 500, 15], [120, 800, 30], [80, 700, 15]])
# Labels: 1 = Approved, 0 = Rejected
y = np.array([0, 1, 0, 1, 1])

model = DecisionTreeClassifier() # Can swap with LogisticRegression()
model.fit(X, y)

print("--- Loan Approval Predictor ---")
income = float(input("Enter Income (in thousands, e.g., 60): "))
credit = float(input("Enter Credit Score (e.g., 720): "))
amount = float(input("Enter Loan Amount (in thousands, e.g., 15): "))

prediction = model.predict([[income, credit, amount]])[0]
print("\nResult: ✅ Approved" if prediction == 1 else "\nResult: ❌ Rejected")