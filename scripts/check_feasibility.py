import os
from highspy import Highs

def read_lp_txt(path):
    """Read LP problem from TXT file in custom format."""
    with open(path) as f:
        lines = [list(map(float, line.strip().split())) for line in f if line.strip()]

    # First line: m, n
    m, n = map(int, lines[0])
    # Next m lines: A matrix
    A = [lines[i+1] for i in range(m)]
    # Next line: b vector
    b = lines[1+m]
    # Next line: c vector
    c = lines[2+m]
    return A, b, c


def has_bfs(A, b, c):
    """Check if LP has a basic feasible solution (feasible Ax ≤ b, x ≥ 0)."""
    m, n = len(A), len(A[0])
    highs = Highs()
    highs.addCols(c=c, lower=[0]*n, upper=[1e9]*n)
    for i in range(m):
        highs.addRow(lower=-1e9, upper=b[i], acoeffs=A[i], astart=None, aend=None)
    highs.setOptionValue("output_flag", False)
    highs.run()
    status = highs.getModelStatus()
    # ModelStatus: 1=Optimal, 2=Feasible, 9=Infeasible
    return status in [1, 2]


def main():
    base_dir = "test"
    total = 0
    feasible = 0

    for subdir, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                total += 1
                path = os.path.join(subdir, file)
                try:
                    A, b, c = read_lp_txt(path)
                    if has_bfs(A, b, c):
                        print(f"✅ {path}: feasible (has BFS)")
                        feasible += 1
                    else:
                        print(f"❌ {path}: infeasible")
                except Exception as e:
                    print(f"⚠️ {path}: error reading ({e})")

    print(f"\nSummary: {feasible}/{total} problems are feasible (have BFS).")

if __name__ == "__main__":
    main()
