import cupy as cp
from cupyx.profiler import benchmark
import matplotlib.pyplot as plt
import numpy as np



batch_sizes = np.floor(np.logspace(start=3, stop=7, num=15)).astype(np.int64)

all_times_numpy = []
all_times_cupy = []
for batch_size in batch_sizes:
    batch_size = int(batch_size)
    np_time = benchmark(np.random.choice, (10 * batch_size, batch_size, False), n_repeat=10)
    cp_time = benchmark(cp.random.choice, (10 * batch_size, batch_size, False), n_repeat=10)
    all_times_numpy.append(np.mean(np_time.cpu_times) + np.mean(np_time.gpu_times))
    all_times_cupy.append(np.mean(cp_time.cpu_times) + np.mean(cp_time.gpu_times))

fig, axs = plt.subplots()
axs.plot(batch_sizes, all_times_numpy, label="numpy")
axs.plot(batch_sizes, all_times_cupy, label="cupy")
axs.legend()
axs.set_xscale("log")
axs.set_yscale("log")

plt.show()
