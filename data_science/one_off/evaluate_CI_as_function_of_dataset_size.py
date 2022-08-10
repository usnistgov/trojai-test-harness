# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import numpy as np

from leaderboard import metrics

x_vals = list()
y_vals = list()

for n in range(10, 1001):
    for k in range(10):
        targets = (np.random.rand(n, 1) > 0.5).astype(np.float32)
        pred = np.random.rand(n, 1)

        ce = metrics.elementwise_binary_cross_entropy(pred, targets)
        ci = metrics.cross_entropy_confidence_interval(ce)
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
