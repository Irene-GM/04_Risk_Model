import numpy as np

path_in = r"F:\acies\workspace\Special\IJHG\00_data\versions\country_features\v12\nl_risk_features_v1.12.csv"

metacols = range(0, 3)
datacols = range(3, 25)
predcols = range(3, 24)

meta_m = np.loadtxt(path_in, delimiter=";", usecols=metacols, skiprows=1)
m = np.loadtxt(path_in, delimiter=";", usecols=datacols, skiprows=1, dtype=np.float32)

print(m.shape)
print(np.unique(m[:,0], return_counts=True))

v30 = len(np.where(m[:,0]>30)[0])
v20 = len(np.where(m[:,0]>20)[0])

print(v20)
print(v30)

n = m.shape[0]

x30 = np.divide(100 * v30, n)
x20 = np.divide(100 * v20, n)

print(x20)
print(x30)

s = m[:,0]
print(np.std(s))
