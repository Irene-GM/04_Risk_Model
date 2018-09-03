import numpy as np


################
# Main program #
################

labels = "dbuiltup;dforest;drecreation;dbrr;dwrl;dwrn;dwrr;dcamping;dcaravan;dcross;dgolf;dheem;dhaven;dsafari;dwater;lu;lc;attr;dbath;meanhaz;stdhaz;exp".split(";")
path_in = r"/home/irene/PycharmProjects/04_Risk_Model/data/no_split/500m/nl_features_v1.12.csv"
# path_pred = r"M:\Documents\workspace\Special\IJHG\data\versions\country_features\v11\nl_centroids_v1.11.csv"
# path_out_tif = r"D:\GeoData\workspaceimg\Special\04_Risk_Model\NL_TB_Risk_v1.12.tif"

# Loading data for train/test
meta_m = np.loadtxt(path_in, delimiter=";", usecols=range(0, 3), skiprows=1)
data_m = np.loadtxt(path_in, delimiter=";", usecols=range(3, 22), skiprows=1)
Y = data_m[:, 0]
X = data_m[:, 1:]
X[np.isnan(X)] = 0

# Loading data for prediction (no target here)
ignore_target_column = 3
meta_p = np.loadtxt(path_in, delimiter=";", usecols=range(0, ignore_target_column), skiprows=1)
data_p = np.loadtxt(path_in, delimiter=";", usecols=range(ignore_target_column+1, 22), skiprows=1)
Xp = data_p
Xp[np.isnan(Xp)] = 0


meta_tr, meta_te, xtrain, ytrain, xtest, ytest = shuffle_split_balance_samples(meta_m, Ycl, Xnz, samples_per_class)

print("Total samples: ", X.shape, Y.shape)
print("Non-zeros: ", Xnz.shape, Ynz.shape)
print("Training with: ", xtrain.shape, ytrain.shape)
print("Testing with: ", xtest.shape, ytest.shape)

# Running models
clf = SVC()
clf.fit(xtrain, ytrain)
ypred = clf.predict(xtest)

# Obtain metrics and FI
oa_sco = round(accuracy_score(ytest, ypred), 4)
ka_sco = round(cohen_kappa_score(ytest, ypred), 4)
print("OA: ", oa_sco)
print("Confusion matrix \n", confusion_matrix(ytest, ypred))

plt.subplot(2, 3, i+1)
cm = confusion_matrix(ytest, ypred)
textstr = "SPC: {0}, OA: {1}, KA: {2}".format(samples_per_class, round(oa_sco, 2), round(ka_sco,2))
plt.title(textstr, size=24)
# plt.matshow(cm, cmap=plt.cm.YlOrBr)
# plt.colorbar()
# plt.show()
sb.heatmap(cm, annot=True, vmin=0, vmax=500, cmap=plt.cm.Blues, annot_kws={"size": 20}, fmt='.0f' )
i += 1
print()