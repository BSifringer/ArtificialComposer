import numpy as np
from neupy.datasets import make_reber
import matplotlib.pyplot as plt
import matplotlib

n = [100, 200,500, 1000,2000,5000, 10000, 20000, 50000, 100000, 500000, 1000000, 5000000, 10000000]

x = np.zeros(len(n))

for j, u in enumerate(n):
	data = make_reber(u)
	for word in data:
		x[j] += len(word)
	x[j] /= u

plt.semilogx(n,abs(x-6))
plt.xlabel('N reber words')
plt.ylabel(' err = |average - 6|')
plt.title('Convergence of average reber word length')
matplotlib.rcParams.update({'font.size': 12})
plt.show()