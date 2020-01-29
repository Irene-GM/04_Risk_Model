import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from sklearn.metrics import mean_squared_error

def samples_per_leaf_node(leaves, xtrain, ytrain):
    t = 0
    dic = defaultdict(list)
    for tree_node_list in leaves.T:
        j = 0
        for node in tree_node_list:
            key = (t, node)
            row_j = [ytrain[j]] + xtrain[j, :].tolist()
            dic[key].append(row_j)
            j += 1
        t += 1
    return dic


def tiny_poisson(l):
    mean_pred, ppf_obs, poi_mod = [None for i in range(3)]
    xtr = np.array([item[1:] for item in l])
    ytr = np.array([item[0] for item in l]).reshape(-1, 1)
    try:
        poi_mod = sm.Poisson(ytr, xtr).fit()
        mean_pred = poi_mod.predict(xtr)  # or use a new x
        sf_obs = stats.poisson.sf(2 - 1, mean_pred)  # average over x in sample
        pmf_obs = stats.poisson.pmf(2, mean_pred)
        ppf_obs = stats.poisson.ppf(q=0.95, mu=mean_pred)  # average over x in sample
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            print("Ignored a singular matrix.")
    return [poi_mod, mean_pred, ppf_obs]

def loop_poisson(dic):
    dicpred = defaultdict(tuple)
    for key in sorted(dic.keys()):
        print(key, len(dic[key]))
        poi_mod, mean_pred, prob_to_obs = tiny_poisson(dic[key])
        if poi_mod != None:
            dicpred[key] = (poi_mod, mean_pred, prob_to_obs)
    return dicpred


################
# Main program #
################

path_in = r"/media/irene/MyPassport/acies/workspace/Special/IJHG/data/versions/country_features/v12/nl_risk_features_v1.12.csv"

metacols = range(0, 3)
datacols = range(3, 25)

meta_m = np.loadtxt(path_in, delimiter=";", usecols=metacols, skiprows=1)
m = np.loadtxt(path_in, delimiter=";", usecols=datacols, skiprows=1)

Y = m[:10000,0]
X = m[:10000,1:]

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.60, random_state=42)

ensemble = RandomForestRegressor(n_estimators=10, min_samples_split=500, min_samples_leaf=500)
ensemble.fit(xtrain, ytrain)

print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

leaves = ensemble.apply(xtrain)

dic = samples_per_leaf_node(leaves, xtrain, ytrain)

dicpred = loop_poisson(dic)

l = []

for key in sorted(dicpred.keys()):
    model = dicpred[key][0]
    mean = dicpred[key][1]
    preds = dicpred[key][2]

    ypred_mean = model.predict(xtest)  # or use a new x
    ypred_ppf = stats.poisson.ppf(q=0.95, mu=ypred_mean)  # average over x in sample
    print(key, "\t\t\t model params?", model.params)

    # plt.hist(ypred_mean, bins=50)
    # plt.hist(ypred_ppf,bins=50)
    # plt.show()
    # print(key, ypred.shape, ypred)
    # plt.hist(ypred, bins=50)
    # plt.show()
    l.append(ypred_ppf)

# for attr in dir(model):
#     if not attr.startswith('_'):
#         print(attr)
#
# print(model.fittedvalues)

arr = np.array(l).T
print(arr.shape)

print(np.unique(arr, return_counts=True))


# plt.imshow(arr, interpolation="None")
# plt.colorbar()
# plt.show()
# avg = np.mean(arr, axis=1).reshape(-1, 1)
# print(avg.shape)
#
# plt.hist(avg, bins=50)
# plt.show()
#
# rmse = np.sqrt(mean_squared_error(ytest, avg))
# print(rmse)

#
# print(ensemble.feature_importances_)

