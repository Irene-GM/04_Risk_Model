import numpy as np
import matplotlib.pyplot as plt

path_in = r"/home/irene/PycharmProjects/04_Risk_Model/data/out/NL_TB_pixel_500m.csv"

meta_m = np.loadtxt(path_in, delimiter=";", usecols=range(0, 3), skiprows=1)
m = np.loadtxt(path_in, delimiter=";", usecols=(3, ), skiprows=1, dtype=np.float32)

# m[np.isnan(m)]=0

print(meta_m.shape, m.shape)

print(np.isnan(m).any(), np.isinf(m).any(), np.isneginf(m).any())

plt.hist(m.tolist(), bins=100)
plt.show()