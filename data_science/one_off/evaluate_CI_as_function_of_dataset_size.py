import numpy as np

from actor_executor import ground_truth

x_vals = list()
y_vals = list()

for n in range(10, 1001):
    for k in range(10):
        targets = (np.random.rand(n, 1) > 0.5).astype(np.float32)
        pred = np.random.rand(n, 1)

        ce = ground_truth.binary_cross_entropy(pred, targets)
        ci = ground_truth.cross_entropy_confidence_interval(ce)
        x_vals.append(n)
        y_vals.append(ci)

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(16, 9), dpi=200)
ax = plt.gca()
x_vals = np.asarray(x_vals)
y_vals = np.asarray(y_vals)
ax.scatter(x_vals, y_vals, c='b', s=4, alpha=0.1)
# ax.set_yscale('log')
ax.set_xlabel('Number Data Points')
ax.set_ylabel('Cross Entropy 95% Confidence Interval')
plt.title('Cross Entropy 95% Confidence Interval as a Function of Data Size')
plt.savefig('CE_95_CI.png')
