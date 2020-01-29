import scipy
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from statsmodels.discrete.count_model import ZeroInflatedPoisson
from statsmodels.genmod import families

################
# Main program #
################

path_in = r"/media/irene/MyPassport/acies/workspace/Special/IJHG/data/versions/country_features/v12/nl_risk_features_v1.12.csv"

# rowid;latitude;longitude;target;dbuiltup;dforest;drecreation;dbrr;dwrl;dwrn;dwrr;dcamping;dcaravan;dcross;dgolf;dheem;dhaven;dsafari;dwater;attr;dbath;lu;lc;maxmeanhaz;maxstdhaz

metacols = range(0, 3)
datacols = range(3, 25)

meta_m = np.loadtxt(path_in, delimiter=";", usecols=metacols, skiprows=1)
m = np.loadtxt(path_in, delimiter=";", usecols=datacols, skiprows=1)

print(meta_m.shape, m.shape)

Y = m[:,0]
X = m[:,1:]

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.60, random_state=42)

poi_model = sm.GLM(ytrain, xtrain, family=sm.families.Poisson())
poi_results = poi_model.fit()
paramet = poi_results.params
ypred = poi_results.predict(paramet, xtest, linear=False)

print(ypred)

# print(ytest)
# print(ypred)
# print(np.unique(ypred, return_counts=True))
#
# clf = ZeroInflatedPoisson(endog = ytrain,exog = xtrain).fit()
#
# ypred = clf.predict(clf.params, xtest)
#
# print(ypred[:100])

# plt.hist(ytest, bins=100)
# plt.hist(ypred, bins=100)
# plt.show()
#

# counts = scipy.stats.gamma.rvs(ypred+1)
# print(counts)

# print(ytest.shape, ypred.shape)
#
# r2 = r2_score(ytest, ypred)
# print("R2: ", r2)
#
# print(np.unique(ypred, return_counts=True))
#
# plt.plot(ytest, ytest, "-")
# plt.plot(ytest, ypred, "o")
# plt.show()

