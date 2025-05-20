# 🧠 Multi-Objective Machining Optimization using TensorFlow & NSGA-II

An ongoing project demonstrates how to **optimize machining process parameters** using a **hybrid Machine Learning + Evolutionary Algorithm** approach. We use **TensorFlow** to approximate the machining process with a neural network, and then apply **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** to find Pareto-optimal solutions that minimize:

- 🔧 **Cutting Force (Force-X)**  
- 🌡️ **Tool Temperature**  
- 📈 **Tool Stress**

---

## 🔍 Problem Statement

Given experimental or simulated data containing:

- **Inputs (decision variables)**:
  - `Feed f_c` (mm/rev)
  - `Cutting Speed v_c` (m/min)
  - `Flank Angle (Alpha)` (°)
  - `Rake Angle (Gamma)` (°)

- **Outputs (objectives to minimize)**:
  - `Force-X (N)`
  - `Temperature (C)`
  - `Tool Stress (MPa)`

We aim to **train a surrogate model** using a neural network to predict outputs for any input configuration, then **optimize the inputs** using the NSGA-II algorithm.

---

## 🛠️ Tools & Libraries Used

- **Python 3.10+**
- [TensorFlow](https://www.tensorflow.org/) (for surrogate modeling)
- [pymoo](https://pymoo.org/) (for NSGA-II optimization)
- **pandas, numpy, scikit-learn** (for preprocessing)
- **matplotlib** (for visualization)

---

## ⚙️ How It Works

### 1. **Data Preprocessing**
- Load data from `summary.csv`
- Normalize input and output features using `MinMaxScaler`

### 2. **Train Neural Network**
- Multi-output regression using TensorFlow
- Input: 4 machining parameters  
- Output: 3 machining objectives (force, temp, stress)

### 3. **Define Optimization Problem**
- Custom class inherits from `pymoo.core.problem.ElementwiseProblem`
- During evaluation, inputs are passed to the trained neural network
- Predicted outputs are used as objective values

### 4. **Run NSGA-II**
- Optimizes all three objectives simultaneously
- Returns a Pareto front of optimal trade-off solutions

### 5. **Visualization**
- A 3D scatter plot displays the Pareto front
- The top solutions are printed and can be saved/exported

---

## 📊 Example Output

- 🎯 Optimal machining configurations that minimize all three target objectives
- 📉 3D Pareto front showing trade-offs between Force, Temp, and Stress
- 📝 Table of top 10 optimal solutions

---

## 📁 Project Structure

