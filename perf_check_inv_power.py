import numpy as np
import time
from helpers import build_progress_bar, inverse_power_law, ou_drift


N = 200
x = np.random.randn(N)


t0 = time.perf_counter()
for _ in range(10000):
	diffs = x[:, None] - x[None, :]
	inverse_power_law(diffs, eps=1e-6, p=0.4).mean(axis=1)
t1 = time.perf_counter()

print(f"Took {t1 - t0} seconds")