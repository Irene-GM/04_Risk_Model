import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


################
# Main program #
################

path_in = r"/home/irene/PycharmProjects/04_Risk_Model/data/poisson_leaves/RF_skewed_leaves_error_metrics_with_zeros_v0.csv"

dic = defaultdict(list)
with open(path_in, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    for row in reader:
        key = (int(row[1]), int(row[0]), row[2])
        val = float(row[3])
        dic[key] = val

lpoi, lnb, lzip, lzinb = [[] for i in range(4)]
all_poi, all_nb, all_zip, all_zinb = [[] for i in range(4)]
for nt in [1, 5, 10, 50, 100]:  # , 5, 10, 50, 100]:
    for ns in range(100, 1500, 100):
        key_poi = (nt, ns, "poi")
        key_nb = (nt, ns, "nb")
        key_zip = (nt, ns, "zip")
        key_zinb = (nt, ns, "zinb")

        val_poi = dic[key_poi]
        val_nb = dic[key_nb]
        val_zip = dic[key_zip]
        val_zinb = dic[key_zinb]

        lpoi.append(val_poi)
        lnb.append(val_nb)
        lzip.append(val_zip)
        lzinb.append(val_zinb)

        print(key_poi, val_poi)
        print(key_nb, val_nb)
        print(key_zip, val_zip)
        print(key_zinb, val_zinb)

    all_poi.append(lpoi)
    all_nb.append(lnb)
    all_zip.append(lzip)
    all_zinb.append(lzinb)

    lpoi, lnb, lzip, lzinb = [[] for i in range(4)]

nrows = len(all_poi)
ncols = 4

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)

cols = ['Model: {0}'.format(mod) for mod in ["Poisson", "Neg. Bin.", "Zero-Inflated Poi.", "Zero-Inflated Neg. Bin."]]
rows = ['{0}'.format(nt) for nt in ['T=1', 'T=5', 'T=10', 'T=50', "T=100"]]

for axis, col in zip(ax[0], cols):
    axis.set_title(col, size=18)

for axis, row in zip(ax[:,0], rows):
    axis.set_ylabel(row, rotation=0, size=18, labelpad=50)

print(len(all_poi))
print(len(all_poi[0]))
for i in range(nrows):
    print("Row: ", i)
    ax[i, 0].plot(all_poi[i])
    ax[i, 1].plot(all_nb[i])
    ax[i, 2].plot(all_zip[i])
    ax[i, 3].plot(all_zinb[i])

    ax[i, 0].grid()
    ax[i, 1].grid()
    ax[i, 2].grid()
    ax[i, 3].grid()

    ax[i, 0].set_ylim(0, 200)
    ax[i, 1].set_ylim(0, 200)
    ax[i, 2].set_ylim(0, 200)
    ax[i, 3].set_ylim(0, 200)

plt.show()



# xlinspace = np.linspace(0, len(all_poi)-1, len(all_poi))
# plt.plot(xlinspace, all_poi)
# plt.plot(xlinspace, all_nb)
# plt.plot(xlinspace, all_zip)
# plt.plot(xlinspace, all_zinb)
# plt.show()
#
#




