import numpy as np
import skill_metrics as sm
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.stats import skew, spearmanr, kendalltau, rankdata

#
# path_in = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/prediction_nl_four_models.csv"
# path_true = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/prediction_nl_true_values.csv"
#
# meta_m = np.loadtxt(path_in, delimiter=";", usecols=range(0, 3))
# m = np.loadtxt(path_in, delimiter=";", usecols=range(3, 8))
# t = np.loadtxt(path_true, delimiter=";", usecols=(3,))
#
# labels = ['Non-Dimensional Observation', 'POI', 'NB', 'ZIP', "ZINB", "RF"]
#
# pred_poi = m[:,0]
# pred_nb = m[:,1]
# pred_zip = m[:,2]
# pred_zinb = m[:,3]
# pred_rf = m[:,4]
#
# target_mod_poi = sm.target_statistics(pred_poi, t, 'data')
# target_mod_nb = sm.target_statistics(pred_nb, t, 'data')
# target_mod_zip = sm.target_statistics(pred_zip, t, 'data')
# target_mod_zinb = sm.target_statistics(pred_zinb, t, 'data')
# target_mod_rf = sm.target_statistics(pred_rf, t, 'data')
#
# taylor_mod_poi = sm.taylor_statistics(pred_poi, t, 'data')
# taylor_mod_nb = sm.taylor_statistics(pred_nb, t, 'data')
# taylor_mod_zip = sm.taylor_statistics(pred_zip, t, 'data')
# taylor_mod_zinb = sm.taylor_statistics(pred_zinb, t, 'data')
# taylor_mod_rf = sm.taylor_statistics(pred_rf, t, 'data')
#
# target_bias = np.array([target_mod_poi['bias'], target_mod_nb['bias'], target_mod_zip['bias'], target_mod_zinb["bias"], target_mod_rf["bias"]])
# target_crmsd = np.array([target_mod_poi['crmsd'], target_mod_nb['crmsd'], target_mod_zip['crmsd'], target_mod_zinb["crmsd"], target_mod_rf["crmsd"]])
# target_rmsd = np.array([target_mod_poi['rmsd'], target_mod_nb['rmsd'], target_mod_zip['rmsd'], target_mod_zinb["rmsd"], target_mod_rf["rmsd"]])
#
# taylor_stdev = np.array([taylor_mod_poi['sdev'][0], taylor_mod_nb['sdev'][1], taylor_mod_zip['sdev'][1], taylor_mod_zinb['sdev'][1], taylor_mod_rf['sdev'][1]])
# taylor_crmsd = np.array([taylor_mod_poi['crmsd'][0], taylor_mod_nb['crmsd'][1], taylor_mod_zip['crmsd'][1], taylor_mod_zinb['crmsd'][1], taylor_mod_rf['crmsd'][1]])
# taylor_ccoef = np.array([taylor_mod_poi['ccoef'][0], taylor_mod_nb['ccoef'][1], taylor_mod_zip['ccoef'][1], taylor_mod_zinb['ccoef'][1], taylor_mod_rf['ccoef'][1]])

# sm.target_diagram(target_bias, target_crmsd, target_rmsd)

# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (15, 5),
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large'}


# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (15, 5),
#           "font.size": 24,
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large'}
#
# pylab.rcParams.update(params)
#
#
# # plt.rcParams.update({'font.size': 26})
# # sm.taylor_diagram(taylor_stdev, taylor_crmsd, taylor_ccoef, markerSize=18, markerColor="indigo", markerLabel=labels, markerLabelColor='black', colCOR='k', colSTD='k', colRMS='darkmagenta', )
# # plt.xlabel('xlabel', fontsize=50)
# # plt.ylabel('ylabel', fontsize='medium')
#
# sm.target_diagram(target_bias,target_crmsd, target_rmsd, markerLabel = labels[1:])


# def rmsle(real, predicted):
#     sum=0.0
#     for x in range(len(predicted)):
#         if predicted[x]<0 or real[x]<0: #check for negative values
#             continue
#         p = np.log(predicted[x]+1)
#         r = np.log(real[x]+1)
#         sum = sum + (p - r)**2
#     return (sum/len(predicted))**0.5

def rmsle(y, h):
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

################
# Main program #
################

path_in = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/to_keep_for_taylor/testing_SPL{0}_NESTI{1}.csv"

n_estimators = [5, 10, 20]
samples_per_leaf = [1000, 1200]
labels = ['Non-Dimensional Observation', 'POI', 'NB', 'ZIP', "ZINB", "RF"]

basename = "{0}_{1}S_{2}T"

labelsize = 16
plt.rcParams['xtick.labelsize'] = labelsize
plt.rcParams['ytick.labelsize'] = labelsize

tit2 = "Predicted distributions from above in function of the number of samples per leaf node"
plt.suptitle(tit2, size=20)

l_stdev, l_crmsd, l_ccoef_sp, l_ccoef_ke, l_names = [[] for i in range(5)]
for i in range(len(samples_per_leaf)):
    for j in range(len(n_estimators)):
        print("SPL: {0}, NESTI: {1}".format(samples_per_leaf[i], n_estimators[j]))
        path_cur = path_in.format(samples_per_leaf[i], n_estimators[j])
        meta_arr = np.loadtxt(path_cur, delimiter=";", usecols=range(0, 3))
        arr = np.loadtxt(path_cur, delimiter=";", usecols=range(3, 9))

        trueval = arr[:, 0]
        pred_poi = arr[:, 1]
        pred_nb = arr[:, 2]
        pred_zip = arr[:, 3]
        pred_zinb = arr[:, 4]
        pred_rf = arr[:, 5]

        trueval_ran = rankdata(sorted(trueval))
        pred_poi_ran = rankdata(sorted(pred_poi))
        pred_nb_ran = rankdata(sorted(pred_nb))
        pred_zip_ran = rankdata(sorted(pred_zip))
        pred_zinb_ran = rankdata(sorted(pred_zinb))
        pred_rf_ran = rankdata(sorted(pred_rf))

        taylor_mod_poi = sm.taylor_statistics_ranked(pred_poi_ran, trueval_ran, pred_poi, trueval, 'data')
        taylor_mod_nb = sm.taylor_statistics_ranked(pred_nb_ran, trueval_ran, pred_nb, trueval, 'data')
        taylor_mod_zip = sm.taylor_statistics_ranked(pred_zip_ran, trueval_ran, pred_nb, trueval, 'data')
        taylor_mod_zinb = sm.taylor_statistics_ranked(pred_zinb_ran, trueval_ran, pred_nb, trueval, 'data')
        taylor_mod_rf = sm.taylor_statistics_ranked(pred_rf_ran, trueval_ran, pred_nb, trueval, 'data')

        l_stdev = l_stdev + [taylor_mod_poi['sdev'][1], taylor_mod_nb['sdev'][1], taylor_mod_zip['sdev'][1], taylor_mod_zinb['sdev'][1], taylor_mod_rf['sdev'][1]]
        l_crmsd = l_crmsd + [taylor_mod_poi['crmlsd'][1], taylor_mod_nb['crmlsd'][1], taylor_mod_zip['crmlsd'][1], taylor_mod_zinb['crmlsd'][1], taylor_mod_rf['crmlsd'][1]]
        l_ccoef_sp = l_ccoef_sp + [taylor_mod_poi['ccoef_sp'][0], taylor_mod_nb['ccoef_sp'][0], taylor_mod_zip['ccoef_sp'][0], taylor_mod_zinb['ccoef_sp'][0], taylor_mod_rf['ccoef_sp'][0]]
        l_ccoef_ke = l_ccoef_ke + [taylor_mod_poi['ccoef_ke'][0], taylor_mod_nb['ccoef_ke'][0], taylor_mod_zip['ccoef_ke'][0], taylor_mod_zinb['ccoef_ke'][0], taylor_mod_rf['ccoef_ke'][0]]
        print(l_ccoef_sp)


        l_names = l_names + [
                             basename.format("POI", samples_per_leaf[i], n_estimators[j]),
                             basename.format("NB", samples_per_leaf[i], n_estimators[j]),
                             basename.format("ZIP", samples_per_leaf[i], n_estimators[j]),
                             basename.format("ZINB", samples_per_leaf[i], n_estimators[j]),
                             basename.format("RF", samples_per_leaf[i], n_estimators[j])]

taylor_stdev = np.array(l_stdev)
taylor_crmsd = np.array(l_crmsd)
taylor_ccoef_sp = np.array(l_ccoef_sp)
taylor_ccoef_ke = np.array(l_ccoef_ke)

plt.subplot(1, 2, 1)
sm.taylor_diagram(taylor_stdev, taylor_crmsd, taylor_ccoef_sp, markerSize=18, markerColor="indigo", markerLabel=l_names, markerLabelColor='black', colCOR='k', colSTD='k', colRMS='darkmagenta', )

plt.subplot(1, 2, 2)
sm.taylor_diagram(taylor_stdev, taylor_crmsd, taylor_ccoef_ke, markerSize=18, markerColor="indigo", markerLabel=l_names, markerLabelColor='black', colCOR='k', colSTD='k', colRMS='darkmagenta', )

plt.show()
