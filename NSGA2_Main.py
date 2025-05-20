import pandas as pd
import numpy as np
import tensorflow as tf
import random
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ---- STEP 1: LOAD & PREPROCESS DATA ---- #
df = pd.read_csv("E:/Csv_Tool/summary.csv")

# Define inputs (decision variables) --- These are the 
X_columns = ["Feed f_c", "Cutting Speed v_c", "Flank angle(Alpha)", "Flank length(l_Alpha1)", "Rake angle (Gamma)"] ##After computing NSGA we get pareto optimal solution of these variables

    # Define objectives  (These are the objectives we want to optimize: minimize Force-X, Temperature, Tool Stress; maximize Force-Y)
    #The solution to the objectives below will be nothing but the combination of the inputs which can provide the best output i.e the pareto solutions
Y_columns = ["Force-X (N)", "Temperature (C)", "Tool Stress (MPa)", "Force-Y (N)"]   ##After computing NSGA we get pareto optimal objectives of these variables

X = df[X_columns].values  # Extract input data
Y = df[Y_columns].values  # Extract output data

# Normalize data
X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()

X_normalized = X_scaler.fit_transform(X)
Y_normalized = Y_scaler.fit_transform(Y)

# ---- STEP 2: TRAIN A NEURAL NETWORK MODEL ---- #
model = keras.Sequential([
    keras.layers.Dense(16, activation="relu", input_shape=(5,)),  # Updated input shape to 5
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(4, activation="linear")  # 4 outputs now
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_normalized, Y_normalized, epochs=200, batch_size=8, verbose=1)

# ---- STEP 3: DEFINE OPTIMIZATION PROBLEM ---- #
class MLBasedMachiningOptimization(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=5, n_obj=4, n_constr=0,
                         xl=np.array([0.08, 100, -2, 0.2, -7]),  # limits of "Feed f_c", "Cutting Speed v_c", "Flank angle(Alpha)", "Flank length(l_Alpha1)", "Rake angle (Gamma)"
                         xu=np.array([0.28, 500, 5, 0.7, 7]))

    def _evaluate(self, x, out, *args, **kwargs):
        x_normalized = X_scaler.transform([x])
        y_pred_normalized = model.predict(x_normalized, verbose=0)
        y_pred = Y_scaler.inverse_transform(y_pred_normalized)[0]
        
        # Negate Force-Y (N) for maximization
        y_pred[3] = -y_pred[3]
        
        out["F"] = y_pred

problem = MLBasedMachiningOptimization()

# ---- STEP 4: SET UP NSGA-II ---- #
algorithm = NSGA2(
    pop_size=100,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# ---- STEP 5: RUN OPTIMIZATION ---- #
result = minimize(
    problem=problem,
    algorithm=algorithm,
    termination=('n_gen', 100),
    seed=42,
    verbose=True
)

# Extract Pareto-optimal solutions
pareto_solutions = result.X
pareto_objectives = result.F
pareto_objectives[:, 3] = -pareto_objectives[:, 3]  # Convert back Force-Y to original scale


# ---- STEP 6: DISPLAY & SAVE RESULTS ---- #
optimal_solutions = pd.DataFrame(pareto_solutions, columns=["Feed f_c", "Cutting Speed v_c", "Flank angle(Alpha)", "Flank length(l_Alpha1)","Rake angle (Gamma)"])
optimal_solutions["Force-X (N)"] = pareto_objectives[:, 0]
optimal_solutions["Temperature (C)"] = pareto_objectives[:, 1]
optimal_solutions["Tool Stress (MPa)"] = pareto_objectives[:, 2]
optimal_solutions["Force-Y (N)"] = pareto_objectives[:, 3]

# Save to Excel
optimal_solutions.to_excel("NIKHIL_pareto_solutions.xlsx", index=False)

print(optimal_solutions.tail(10))


# ---- STEP 7: PLOT PARETO FRONT ---- #

#["Force-X (N)", "Temperature (C)", "Tool Stress (MPa)", "Force-Y (N)"]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(pareto_objectives[:, 0], pareto_objectives[:, 3], pareto_objectives[:, 1], c=pareto_objectives[:,2], cmap='viridis', marker="o",alpha=0.8)
ax.set_xlabel("Force-X (N)") ## Force X xaxis
ax.set_ylabel("Force-Y (N)") ## force \y in y axis
ax.set_zlabel("Temperature (C)") #Temp on z axis and add the color scale of tool stress to the enitre graph
ax.set_title("Pareto Front of ML-Based Optimization")

cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label('Tool Stress (MPa)')
plt.show()

