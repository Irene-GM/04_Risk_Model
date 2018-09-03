import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import normaltest, skew, kurtosis, boxcox, gamma
from scipy.stats import norm


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

def trim_zeros(Y, X):
    pos_nze = np.where(Y!=0)
    lx, ly = [[] for i in range(2)]
    for pos in pos_nze:
        ly.append(Y[pos])
        lx.append(X[pos,:])
    Y_ = np.squeeze(np.array(ly), axis=0)
    X_ = np.squeeze(np.array(lx), axis=0)
    print(Y_.shape, X_.shape)
    return [Y_, X_]

def gaussify(Y, k):
    num = np.exp(np.divide(np.power(Y, 2), 2*np.pi*k*k))
    den = np.sqrt(2*np.pi*k)
    return np.divide(num, den)

def gaussify2(Y, alpha, beta, mu):
    # Check: https://stats.stackexchange.com/questions/37461/the-relationship-between-the-gamma-distribution-and-the-normal-distribution

    algo = np.divide(np.abs(Y-mu), alpha)
    term_num = np.power(algo, beta)
    num = beta * np.exp(term_num)

    algo2 = gamma.stats(np.divide(1, beta), moments='m')
    den = 2 * alpha * algo2
    return np.divide(num, den)

def test_normality(Y):
    k2, p = normaltest(Y)
    alpha = 1e-2
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



################
# Main program #
################

labels = "dbuiltup;dforest;drecreation;dbrr;dwrl;dwrn;dwrr;dcamping;dcaravan;dcross;dgolf;dheem;dhaven;dsafari;dwater;lu;lc;attr;dbath;meanhaz;stdhaz;exp".split(";")
path_in = r"D:\UTwente\workspace\Special\04_Risk_Model\data\no_split\nl_risk_features_v1.12b.csv"
# path_pred = r"M:\Documents\workspace\Special\IJHG\data\versions\country_features\v11\nl_centroids_v1.11.csv"
path_out_tif = r"D:\GeoData\workspaceimg\Special\04_Risk_Model\NL_TB_Risk_v1.12.tif"

textstr = '$p-value={0:.3f}$\n$skewness={1:.3f}$\n$kurtosis={2:.3f}$'
titlstr = '$\lambda={0:.1f}$'

# Loading data for train/test
meta_m = np.loadtxt(path_in, delimiter=";", usecols=range(0, 3), skiprows=1)
data_m = np.loadtxt(path_in, delimiter=";", usecols=range(3, 25), skiprows=1)
Y = data_m[:, 0]
X = data_m[:, 1:]
X[np.isnan(X)] = 0

Ynz, Xnz = trim_zeros(Y, X)

# Also known as shape, loc, and scale parameters
# or alpha, loc, beta
fit_shape, fit_loc, fit_scale = gamma.fit(Ynz, floc=0)
print("alpha/shape: {0} \tloc: {1} \t beta/scale: {2}".format(fit_shape, fit_loc, fit_scale))

plt.subplot(2, 2, 1)
plt.title('Histogram on target (skewed, positive samples)')
plt.hist(Ynz, bins=200, color="gray")
plt.grid()

plt.subplot(2, 2, 2)
plt.title("Fitted gamma distribution (pdf)")
x = np.linspace(0, 200, 201)
y = gamma.pdf(x, fit_shape, fit_loc, fit_scale)
plt.plot(x, y, linewidth=3, color="blue")
plt.grid()
plt.ylim(0, 0.25)

plt.subplot(2, 2, 3)
plt.title("Transformations")
Y_gauss = gaussify2(y, fit_shape, fit_scale, fit_loc)
plt.plot(x, Y_gauss)
plt.grid()

plt.subplot(2, 2, 4)
new_mean = fit_shape * fit_scale
new_std = np.sqrt(fit_shape * (fit_scale ** 2))

mu = new_mean
std = new_std

print("Mean: {0} \tStdev: {1}".format(mu, std))

p = norm.pdf(x, mu, std)

test_normality(p[:10])
test_normality(Ynz[:10])


plt.plot(x, p, 'k', linewidth=2)



plt.show()









