import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

path_in = r"/media/irene/MyPassport/acies/workspace/Special/IJHG/data/versions/country_features/v12/nl_risk_features_v1.12.csv"

metacols = range(0, 3)
datacols = range(3, 25)

meta_m = np.loadtxt(path_in, delimiter=";", usecols=metacols, skiprows=1)
m = np.loadtxt(path_in, delimiter=";", usecols=datacols, skiprows=1)

Y = m[:,0]
X = m[:,1:]

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.60, random_state=42)

results = sm.Poisson(ytrain, xtrain).fit()
mean_predicted = results.predict(xtest)   # or use a new x
# plt.hist(mean_predicted, bins=20)
# plt.show()
print(mean_predicted)
sf_2more = stats.poisson.sf(2 - 1, mean_predicted) # average over x in sample
ppf_2more = stats.poisson.ppf(q=0.95, mu=mean_predicted) # average over x in sample
prob_2_obs = stats.poisson.pmf(2, mean_predicted)

xlinspace = np.linspace(0, len(mean_predicted)-1, len(mean_predicted))
plt.subplot(2, 2, 1)
plt.title("SF")
plt.hist(sf_2more, bins=50)
plt.subplot(2, 2, 2)
plt.title("PPF")
plt.hist(ppf_2more, bins=50, color="blue")
plt.subplot(2, 2, 3)
plt.title("PMF (to obs)")
plt.hist(prob_2_obs, bins=50)
plt.subplot(2, 2, 4)
plt.title("Ytest")
plt.hist(ytest, bins=50, color="blue")

plt.show()
