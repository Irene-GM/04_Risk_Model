import gdal
import jenkspy
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import defaultdict
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import normaltest, skew, kurtosis, boxcox
import seaborn as sb
import itertools



def goodness_of_variance_fit(array, classes):
    classes = jenkspy.jenks_breaks(array, nb_class=classes)
    classifiedd = np.array([classify(i, classes) for i in array])
    maxz = np.amax(classifiedd)
    zone_indices = [[idx for idx, val in enumerate(classifiedd) if zone + 1 == val] for zone in range(maxz)]
    sdam = np.sum((array - array.mean()) ** 2)
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]
    sdcm = np.sum([np.sum((cla - cla.mean()) ** 2) for cla in array_sort])
    gvf = (sdam - sdcm) / sdam
    return gvf

def classify(value, breaks):
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1

def natural_breaks(m):
    gvf = 0.0
    nclasses = 2
    print("Calculating breaks")
    while gvf <= 0.8 and nclasses < 10:
        gvf = goodness_of_variance_fit(m, nclasses)
        print("\tGVF for {0} classes: {1}".format(nclasses, gvf))
        nclasses += 1

    print("Running jenks with {0} classes".format(nclasses-1))
    breaks = jenkspy.jenks_breaks(m, nb_class=nclasses-1)
    # breaks = [1, 3, 7, 12, 18, 25, 50]
    print("\t Breaks in: ", breaks)
    classified = np.array([classify(i, breaks) for i in m])
    return classified

def classify_target(Y):
    Ycl = natural_breaks(Y)
    return Ycl


def show_FI(ensemble, headers, lim):
    cols = headers
    i = 1
    print("\nFeature Importances")
    print("-"*40)
    for item in list(reversed(sorted(zip(cols, ensemble.feature_importances_), key=itemgetter(1))))[:lim]:
        print(i, ")\t", item[0], "\t\t", np.round(item[1] * 100, decimals=2), "%")
        i += 1

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

def choose_random_samples_per_class(samples_per_class, lim):
    while True:
        chosen_tr = np.random.randint(0, len(samples_per_class), lim)
        chosen_te = np.random.randint(0, len(samples_per_class), lim)
        if not np.in1d(chosen_tr, chosen_te).any():
            break
    return chosen_tr, chosen_te

def shuffle_rows(meta, Y, X):
    m = np.hstack((meta, Y.reshape(-1, 1), X))
    np.random.shuffle(m)
    meta_ = m[:,0:3]
    Y_ = m[:,3]
    X_ = m[:,4:]
    return [meta_, Y_, X_]

def split_samples(l, samples_per_class):
    if len(l) >= samples_per_class:
        threshold = int(np.divide(len(l) * 66, 100))
    else:
        threshold = int(np.divide(len(l) * 50, 100))

    ltr = l[0:threshold]
    lte = l[threshold:]
    return [ltr, lte]

def shuffle_split_balance_samples(meta, Y, X, samples_per_class):
    dicpos, dicsam, dicspl = [defaultdict(list) for i in range(3)]
    ltr, lte = [[] for i in range(2)]
    classes, counts = np.unique(Y, return_counts=True)

    # avg = int(np.divide(nsamples, len(classes)))

    # Find samples belonging to each class
    print("Finding samples per class")
    for classs in classes:
        dicpos[classs] = np.where(Y==classs)[0]

    # For each class, add to a list its associated samples
    print("Linking samples to a class")
    for key in sorted(dicpos.keys()):
        for idx in dicpos[key]:
            newrow = meta[idx, :].tolist() + [Y[idx]] + X[idx,:].tolist()
            dicsam[key].append(newrow)

    # Shuffle and split the samples per class
    print("Shuffling and splitting data in train/test")
    for key in sorted(dicsam.keys()):
        np.random.shuffle(dicsam[key])
        ltr_i, lte_i = split_samples(dicsam[key], samples_per_class)
        tup = (ltr_i, lte_i)
        dicspl[key] = tup

    # Balance the number of samples per class
    print("Balancing number of samples per class")
    for key in sorted(dicspl.keys()):
        print("\tClass ", key)
        ltr_i = dicspl[key][0]
        lte_i = dicspl[key][1]

        if len(ltr_i) <= samples_per_class:
            thr = len(ltr_i) # Find a pivot, and we simply this chunk
        else:
            thr = samples_per_class

        ltr = ltr + ltr_i[0:thr]
        lte = lte + lte_i[0:thr]
        print("\t\t Train: {0} \t Test: {1}".format(len(ltr_i[0:thr]), len(lte[0:thr])))

    mtr = np.array(ltr)
    mte = np.array(lte)

    meta_tr = mtr[:, 0:3]
    meta_te = mte[:, 0:3]

    Ytr = mtr[:, 3]
    Yte = mte[:, 3]

    Xtr = mtr[:, 4:]
    Xte = mte[:, 4:]

    return [meta_tr, meta_te, Xtr, Ytr, Xte, Yte]


def place_in_array(stack):
    basepath = r"E:/RSData/KNMI/Xdays_v2/tmax/2014_tmax_365.tif"
    tif = gdal.Open(basepath)
    canvas = np.multiply(np.ones((350, 300)), -1)
    geotransform = tif.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]

    for row in stack:
        x = row[1]
        y = row[2]
        v = row[3]
        xoffset = int((x - originX) / pixel_width)
        yoffset = int((y - originY) / pixel_height)
        canvas[yoffset, xoffset] = v

    return canvas

def write_tif(m, path):
    tif_template = gdal.Open("E:/RSData/KNMI/yearly/tmax/2014_tmax.tif")
    rows = tif_template.RasterXSize
    cols = tif_template.RasterYSize

    # Get the origin coordinates for the tif file
    geotransform = tif_template.GetGeoTransform()
    outDs = tif_template.GetDriver().Create(path, rows, cols, 1, gdal.GDT_Float32)
    outBand = outDs.GetRasterBand(1)

    # write the data
    outDs.GetRasterBand(1).WriteArray(m)

    # flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()
    outBand.SetNoDataValue(-99)

    # georeference the image and set the projection
    outDs.SetGeoTransform(geotransform)
    outDs.SetProjection(tif_template.GetProjection())
    outDs = None
    outBand = None

def make_hist(Y):
    plt.grid()
    ax = plt.gca()
    ax.grid(color='gray', linestyle='--', linewidth=2)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    plt.title("Histogram of tick bites per pixel", size=40)
    plt.xlabel("# tb/pixel", size=34)
    plt.ylabel("Frequency", size=34)
    plt.xticks(size=24)
    plt.yticks(size=24)
    plt.hist(Y, bins=60)
    plt.show()




################
# Main program #
################

# 1.4: Contains the balancing of the

labels = "dbuiltup;dforest;drecreation;dbrr;dwrl;dwrn;dwrr;dcamping;dcaravan;dcross;dgolf;dheem;dhaven;dsafari;dwater;lu;lc;attr;dbath;meanhaz;stdhaz;exp".split(";")
path_in = r"D:\UTwente\workspace\Special\04_Risk_Model\data\no_split\nl_risk_features_v1.12b.csv"
# path_pred = r"M:\Documents\workspace\Special\IJHG\data\versions\country_features\v11\nl_centroids_v1.11.csv"
path_out_tif = r"D:\GeoData\workspaceimg\Special\04_Risk_Model\NL_TB_Risk_v1.12.tif"

# Loading data for train/test
meta_m = np.loadtxt(path_in, delimiter=";", usecols=range(0, 3), skiprows=1)
data_m = np.loadtxt(path_in, delimiter=";", usecols=range(3, 25), skiprows=1)
Y = data_m[:, 0]
X = data_m[:, 1:]
X[np.isnan(X)] = 0

# Loading data for prediction (no target here)
ignore_target_column = 3
meta_p = np.loadtxt(path_in, delimiter=";", usecols=range(0, ignore_target_column), skiprows=1)
data_p = np.loadtxt(path_in, delimiter=";", usecols=range(ignore_target_column+1, 25), skiprows=1)
Xp = data_p
Xp[np.isnan(Xp)] = 0

# This is to filter values
# beyond sigma std. dev. from the mean
# Check outlier_detection.py for that
stdev = 25
Y[Y>stdev] = stdev

# Processing train/test data
Ynz, Xnz = trim_zeros(Y, X)
Ycl = classify_target(Ynz.reshape(-1, 1))

i = 0
fig, ax = plt.subplots(nrows=2, ncols=3)
pairs = list(itertools.product(range(0, 3), range(0, 5)))
sb.set(font_scale=1.6)

plt.suptitle("Tick bite risk classification (positive samples, skewed target)", size=24)
for samples_per_class in range(100, 700, 100):
    print("Samples per class: ", samples_per_class)

    meta_tr, meta_te, xtrain, ytrain, xtest, ytest = shuffle_split_balance_samples(meta_m, Ycl, Xnz, samples_per_class)

    print("Total samples: ", X.shape, Y.shape)
    print("Non-zeros: ", Xnz.shape, Ynz.shape)
    print("Training with: ", xtrain.shape, ytrain.shape)
    print("Testing with: ", xtest.shape, ytest.shape)

    # Running models
    clf = RandomForestClassifier(n_estimators=100)
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

plt.show()


