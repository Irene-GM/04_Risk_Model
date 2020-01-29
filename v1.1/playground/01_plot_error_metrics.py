import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

path_err = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/short_run_4mod_v1.0_error_metrics_NESTI_10.csv"

spl = [200, 400, 600, 1000, 1200]
n_esti = [1]

l = []
with open(path_err, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    for row in reader:
        l.append(row)

lpoi, lnb, lzip, lzinb, lrf = [[] for i in range(5)]
for row in l:
    if "poi" in row:
        lpoi.append(row)
    elif "nb" in row:
        lnb.append(row)
    elif "zip" in row:
        lzip.append(row)
    elif "zinb" in row:
        lzinb.append(row)
    else:
        lrf.append(row)


colors = ["royalblue", "rebeccapurple", "goldenrod", "olivedrab", "maroon"]
labels = ["POI", "NB", "ZIP", "ZINB", "RF"]
lbl_spl = ["200", "400", "600", "1000", "1200"]

fig, ax = plt.subplots(nrows=3, ncols=6, figsize=(10, 3))

fig.subplots_adjust(wspace=0.2, hspace=0.2)

size_title = 20
size_xlbl = 16
i = 0
for l in [lpoi, lnb, lzip, lzinb, lrf]:

    lr2 = [float(item[3]) for item in l]
    lrmse = [float(item[4]) for item in l]
    lrmsep = [float(item[5]) for item in l]
    lmape = [float(item[6]) for item in l]
    lsmape = [float(item[7]) for item in l]

    ax[0, 0].set_title("R2", size=size_title)
    ax[0, 0].grid()
    ax[0, 0].plot(lr2, "-", linewidth=2, color=colors[i], label=labels[i])
    ax[0, 0].legend()
    ax[0, 0].set_xlabel("SPL", size=size_xlbl)

    ax[0, 1].set_title("RMSE", size=size_title)
    ax[0, 1].grid()
    ax[0, 1].plot(lrmse, "-", linewidth=2, color=colors[i], label=labels[i])
    ax[0, 1].legend()
    ax[0, 1].set_xlabel("SPL", size=size_xlbl)

    ax[0, 2].set_title("RMSE-P", size=size_title)
    ax[0, 2].grid()
    ax[0, 2].plot(lrmsep, "-", linewidth=2, color=colors[i], label=labels[i])
    ax[0, 2].legend()
    ax[0, 2].set_xlabel("SPL", size=size_xlbl)

    ax[0, 3].set_title("MAPE", size=size_title)
    ax[0, 3].grid()
    ax[0, 3].plot(lmape, "-", linewidth=2, color=colors[i], label=labels[i])
    ax[0, 3].legend()
    ax[0, 3].set_xlabel("SPL", size=size_xlbl)

    ax[0, 4].set_title("SMAPE", size=size_title)
    ax[0, 4].grid()
    ax[0, 4].plot(lsmape, "-",  linewidth=2, color=colors[i], label=labels[i])
    ax[0, 4].legend()
    ax[0, 4].set_xlabel("SPL", size=size_xlbl)

    ax[0, 5].set_title("AIC", size=size_title)
    ax[0, 5].grid()
    ax[0, 5].plot(lsmape, "-",  linewidth=2, color=colors[i], label=labels[i])
    ax[0, 5].legend()
    ax[0, 5].set_xlabel("SPL", size=size_xlbl)

    i += 1

plt.show()


