import os
import numpy as np
from highspy import Highs

# parameters
num_problems = 20
folder = "test"
os.makedirs(folder, exist_ok=True)

def save_lp(folder, A, b, c, problem_id):
    """Save LP to txt files in the given folder."""
    m, n = A.shape
    os.makedirs(folder, exist_ok=True)
    # A.txt
    with open(os.path.join(folder, f"A_{problem_id}.txt"), "w") as f:
        f.write(f"{m} {n}\n")
        for row in A:
            f.write(" ".join(map(str, row)) + "\n")
        f.write(" ".join(map(str, b)) + "\n")
        f.write(" ".join(map(str, c)) + "\n")
    

def check_feasible(A, b, c):
    """Check feasibility with HiGHS."""
    m, n = A.shape
    highs = Highs()
    # Add columns with costs, bounds, and names
    highs.addCols(
        n,                          # number of columns
        c.astype(np.float64),      # objective coefficients
        np.zeros(n, dtype=np.float64),  # lower bounds
        np.full(n, 1e9, dtype=np.float64),  # upper bounds
        0,                          # number of nonzeros in matrix
        np.array([], dtype=np.int32),  # row indices
        np.array([], dtype=np.int32),  # column starts
        np.array([], dtype=np.float64)  # element values
    )
    # Add constraints row by row
    for i in range(m):
        indices = np.arange(n, dtype=np.int32)
        starts = np.array([0], dtype=np.int32)
        values = A[i].astype(np.float64)
        highs.addRows(
            1,                      # number of rows
            np.array([-1e9], dtype=np.float64),  # lower bounds
            np.array([b[i]], dtype=np.float64),  # upper bounds
            n,                      # number of nonzeros
            indices,                # column indices
            starts,                 # row starts
            values                  # element values
        )
    highs.setOptionValue("output_flag", False)
    highs.run()
    return highs.getModelStatus() in [1, 2]  # 1=optimal, 2=feasible

# generator loop
generated = 0
tries = 0
while generated < num_problems and tries < 1000:
    tries += 1
    m = np.random.randint(2, 5)   # 2–4 constraints
    n = np.random.randint(3, 6)   # 3–5 variables
    A = np.random.randint(1, 6, size=(m, n))
    x_true = np.random.randint(1, 5, size=n)
    b = A @ x_true
    c = np.random.randint(1, 10, size=n)

    if True:
        prob_folder = os.path.join(folder, f"problem_{generated+1:02d}")
        save_lp(prob_folder, A, b, c, generated)
        generated += 1
        print(f"✅ Saved {prob_folder}")
    else:
        print("Skipping infeasible problem")

print(f"\nGenerated {generated} feasible LP problems in '{folder}/'")

