import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import normaltest, skew, kurtosis, boxcox
import matplotlib.gridspec as gridspec

def test_normality(Y):
    k2, p = normaltest(Y)
    alpha = 1e-3
    print("\tp = {:g}".format(p))
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("\tThe null hypothesis can be rejected")
    else:
        print("\tThe null hypothesis cannot be rejected")
    s = skew(Y)
    k = kurtosis(Y)
    print("\tSkew: ", s)
    print("\tKurtosis: ", k)
    print()
    return [p, s, k]


def boxcox_tester(Y):
    l = []
    print("Box-cox Transformations")
    print("-"*50)
    for lmbd in np.arange(-3.5, 3.5, 0.5):
        print("Lambda = {0}".format(lmbd))
        print("------------")
        arr = boxcox(Y, lmbda=lmbd)
        p, s, k = test_normality(arr)
        tup = (lmbd, arr, p, s, k)
        l.append(tup)
    return l

def trim_value(Y, X, v):
    pos_nze = np.where(Y!=v)
    lx, ly = [[] for i in range(2)]
    for pos in pos_nze:
        ly.append(Y[pos])
        lx.append(X[pos,:])
    Y_ = np.squeeze(np.array(ly), axis=0)
    X_ = np.squeeze(np.array(lx), axis=0)
    print(Y_.shape, X_.shape)
    return [Y_, X_]


################
# Main program #
################

labels = "dbuiltup;dforest;drecreation;dbrr;dwrl;dwrn;dwrr;dcamping;dcaravan;dcross;dgolf;dheem;dhaven;dsafari;dwater;lu;lc;attr;dbath;meanhaz;stdhaz".split(";")
# path_in = r"D:\UTwente\workspace\Special\04_Risk_Model\data\no_split\nl_risk_features_v1.12b.csv"
# path_pred = r"M:\Documents\workspace\Special\IJHG\data\versions\country_features\v11\nl_centroids_v1.11.csv"
path_in = r"/home/irene/PycharmProjects/04_Risk_Model/data/no_split/500m/nl_features_v1.12.csv"
# path_out_tif = r"D:\GeoData\workspaceimg\Special\04_Risk_Model\NL_TB_Risk_v1.12.tif"

textstr = '$p-value={0:.3f}$\n$skewness={1:.3f}$\n$kurtosis={2:.3f}$'
titlstr = '$\lambda={0:.1f}$'

# Loading data for train/test
meta_m = np.loadtxt(path_in, delimiter=";", usecols=range(0, 3), skiprows=1)
data_m = np.loadtxt(path_in, delimiter=";", usecols=range(3, 22), skiprows=1)
Y = data_m[:, 0]
X = data_m[:, 1:]
X[np.isnan(X)] = 0

Ynz, Xnz = trim_value(Y, X, 0)
Yno, Xno = trim_value(Ynz, Xnz, 1)
Yno, Xno = trim_value(Yno, Xno, 2)
Yno, Xno = trim_value(Yno, Xno, 3)


l = boxcox_tester(Yno)

i = 1
bins = 50
first = True
pairs = list(itertools.product(range(0, 3), range(0, 5)))
fig, ax = plt.subplots(nrows=3, ncols=5)

plt.suptitle("Box-Cox transformations to correct for non-normality (TB >= 2)", size=24)
for tup in l:
    lmbd, Yt, p, s, k = [item for item in tup]
    px, py = pairs[i-1]
    print(px, py)
    ax[px, py].set_ylim(0, 3000)
    ax[px, py].yaxis.grid(which="major", color='gray', linestyle='-', linewidth=1)
    if first == True:
        ax[px, py].set_title("Original target ", size=20)
        ax[px, py].hist(Y, bins=bins, color="#444444")
        first =False
    else:
        ax[px, py].set_title(titlstr.format(lmbd), size=20)
        ax[px, py].hist(Yt, bins=bins)
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax[px, py].text(0.60, 0.97, textstr.format(p, s, k), fontsize=12, verticalalignment='top', bbox=props, transform=ax[px, py].transAxes)
    ax[px, py].tick_params(axis='both', labelsize=12)
    i += 1

ax[-1, -1].axis('off')


plt.show()








# More code below







