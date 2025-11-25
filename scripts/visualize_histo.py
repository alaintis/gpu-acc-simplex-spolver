import matplotlib.pyplot as plt

# Simulate reading from a file (replace with actual path)
data_text = """
../../dphpc-simplex-data/netlib/presolved/blend.presolved
-30.812150,-30.812150,0.000000
6379.179798,137.428385

../../dphpc-simplex-data/netlib/presolved/kb2.presolved
-1749.900130,-1749.900130,0.000000
2412.506875,119.837750

../../dphpc-simplex-data/netlib/presolved/lotfi.presolved
-25.264706,-25.264706,0.000000
18113.451119,123.858319

../../dphpc-simplex-data/netlib/presolved/sc105.presolved
-52.202061,-52.202061,0.000000
1872.652252,101.579232

../../dphpc-simplex-data/netlib/presolved/sc205.presolved
-52.202061,-52.202061,0.000000
4734.475775,118.988437

../../dphpc-simplex-data/netlib/presolved/sc50a.presolved
-64.575077,-64.575077,0.000000
2470.073985,124.714883

../../dphpc-simplex-data/netlib/presolved/sc50b.presolved
-70.000000,-70.000000,0.000000
826.584180,128.465381

../../dphpc-simplex-data/netlib/presolved/scagr7.presolved
-2331389.824333,-2331389.824331,0.000002
5131.917576,139.199792

../../dphpc-simplex-data/netlib/presolved/stocfor1.presolved
-41131.976219,-41131.976219,0.000000
4369.356533,118.892240
"""

lines = [l.strip() for l in data_text.split("\n") if l.strip()]

names = []
deltas = []
cpu_times = []
gpu_times = []

i = 0
while i < len(lines):
    name = lines[i]
    cpu_opt, gpu_opt, delta = map(float, lines[i+1].split(","))
    cpu_t, gpu_t = map(float, lines[i+2].split(","))
    
    # shorten name
    short = name.split("/")[-1].replace(".presolved", "")
    
    names.append(short)
    deltas.append(delta)
    cpu_times.append(cpu_t)
    gpu_times.append(gpu_t)

    i += 3

# Plot
plt.figure(figsize=(12, 6))

x = range(len(names))
width = 0.4

plt.bar([p - width/2 for p in x], cpu_times, width=width, label="CPU Time")
plt.bar([p + width/2 for p in x], gpu_times, width=width, label="CUOPT Time")

plt.xticks(x, [f"{names[i]}\nΔ={deltas[i]:.6f}" for i in range(len(names))], rotation=45, ha="right")
plt.ylabel("Time")
plt.title("CPU vs CUOPT Times per Problem")
plt.legend()
plt.tight_layout()
plt.savefig("histo_times.png", bbox_inches='tight')
plt.show()
