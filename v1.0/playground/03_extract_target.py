import numpy as np

path_in = r"D:\UTwente\workspace\Special\04_Risk_Model\data\no_split\nl_risk_features_v1.12b.csv"
path_out = r"D:\UTwente\workspace\Special\04_Risk_Model\data\no_split\nl_tb_per_pixel.csv"

# Loading data for train/test
meta_m = np.loadtxt(path_in, delimiter=";", usecols=range(0, 4), skiprows=1)

np.savetxt(path_out, meta_m, delimiter=";", fmt="%d")

