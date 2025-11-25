import matplotlib.pyplot as plt
import sys

def load_data(path):
    """
    Reads the file which contains repeated blocks of:
        GPU_time, CUOPT_time
        delta
    Blocks may be separated by blank lines.
    """
    gpu_times = []
    cuopt_times = []
    deltas = []

    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        # first line: GPU, CUOPT
        gpu_t, cuopt_t = map(float, lines[i].split(","))
        # second line: delta
        delta = float(lines[i + 1])

        gpu_times.append(gpu_t)
        cuopt_times.append(cuopt_t)
        deltas.append(delta)

        i += 2

    return gpu_times, cuopt_times, deltas


def plot_data(gpu_times, cuopt_times, deltas):
    n = len(gpu_times)
    x = range(n)

    plt.figure(figsize=(14, 6))

    width = 0.4

    # bars
    plt.bar([xi - width/2 for xi in x], gpu_times, width=width, label="GPU Solver Time")
    plt.bar([xi + width/2 for xi in x], cuopt_times, width=width, label="CUOPT Time")

    # x-axis labels: Problem i and delta beneath
    labels = [f"P{i+1}\nΔ={deltas[i]:.2e}" for i in range(n)]
    plt.xticks(x, labels, rotation=45, ha="right")

    plt.ylabel("Time")
    plt.title("GPU Solver vs CUOPT Time per Problem")
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_times_random_gpu.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Use command line argument if provided, otherwise default filename
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "data.txt"   # default if no argument given

    gpu_times, cuopt_times, deltas = load_data(path)
    plot_data(gpu_times, cuopt_times, deltas)
