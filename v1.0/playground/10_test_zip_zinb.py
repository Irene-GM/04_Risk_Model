import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = np.random.randint(99, size=(800, 21))
Y = np.random.randint(2, size=(800, 1))

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.60, random_state=42)

print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

print("Model: Poisson")
poi_mod = sm.Poisson(ytrain, xtrain).fit(method="newton", maxiter=50)
poi_mean_pred = poi_mod.predict(xtest)
poi_ppf_obs = stats.poisson.ppf(q=0.95, mu=poi_mean_pred)
poi_rmse = np.sqrt(mean_squared_error(ytest, poi_ppf_obs))

print("Model: Neg. Binomial")
nb_mod = sm.NegativeBinomial(ytrain, xtrain).fit(start_params=None, method='newton', maxiter=50)
nb_pred = nb_mod.predict(xtest)
nb_rmse = np.sqrt(mean_squared_error(ytest, nb_pred))

print(np.ones(len(xtest)).shape)

print("Model: Zero Inflated Poisson")
zip_mod = sm.ZeroInflatedPoisson(ytrain, xtrain).fit(method="newton", maxiter=50)
zip_mean_pred = zip_mod.predict(xtest, exog_infl=np.ones((len(xtest), 1)))
zip_ppf_obs = stats.poisson.ppf(q=0.95, mu=zip_mean_pred)
zip_rmse = np.sqrt(mean_squared_error(ytest, zip_ppf_obs))

print("Model: Zero Inflated Neg. Binomial")
zinb_mod = sm.ZeroInflatedNegativeBinomialP(ytrain, xtrain).fit(method="newton", maxiter=50)
zinb_pred = zinb_mod.predict(xtest, exog_infl=np.ones((len(xtest), 1)))
zinb_rmse = np.sqrt(mean_squared_error(ytest, zinb_pred))

print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

print("RMSE Poisson: ", poi_rmse)
print("RMSE Neg. Bin.: ", nb_rmse)
print("RMSE ZIP", zip_rmse)
print("RMSE ZINB: ", zinb_rmse)