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

# path_in = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/to_keep_for_taylor/testing_SPL{0}_NESTI{1}.csv"

path_in = r"D:\UTwente\PycharmProjects\04_Risk_Model\data\skewed_leaves\to_keep_for_taylor/testing_SPL{0}_NESTI{1}.csv"

n_estimators = [10, 20, 50]
samples_per_leaf = [100, 200, 400, 600, 800]
labels = ['Non-Dimensional Observation', 'POI', 'NB', 'ZIP', "ZINB", "RF"]

basename = "{0}S_{1}T"

labelsize = 16
plt.rcParams['xtick.labelsize'] = labelsize
plt.rcParams['ytick.labelsize'] = labelsize

l_stdev_poi, l_crmsd_poi, l_ccoef_sp_poi, l_ccoef_ke_poi, l_names_poi = [[] for i in range(5)]
l_stdev_nb, l_crmsd_nb, l_ccoef_sp_nb, l_ccoef_ke_nb, l_names_nb = [[] for i in range(5)]
l_stdev_zip, l_crmsd_zip, l_ccoef_sp_zip, l_ccoef_ke_zip, l_names_zip = [[] for i in range(5)]
l_stdev_zinb, l_crmsd_zinb, l_ccoef_sp_zinb, l_ccoef_ke_zinb, l_names_zinb = [[] for i in range(5)]
l_stdev_rf, l_crmsd_rf, l_ccoef_sp_rf, l_ccoef_ke_rf, l_names_rf = [[] for i in range(5)]

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

        l_stdev_poi = l_stdev_poi + [taylor_mod_poi['sdev'][1] + np.random.uniform(0,0.10,1)[0]]
        l_stdev_nb = l_stdev_nb + [taylor_mod_nb['sdev'][1] + np.random.uniform(0,0.10,1)[0]]
        l_stdev_zip = l_stdev_zip + [taylor_mod_zip['sdev'][1] + np.random.uniform(0,0.10,1)[0]]
        l_stdev_zinb = l_stdev_zinb + [taylor_mod_zinb['sdev'][1] + np.random.uniform(0,0.10,1)[0]]
        l_stdev_rf = l_stdev_rf + [taylor_mod_rf['sdev'][1] + np.random.uniform(0,0.10,1)[0]]

        l_crmsd_poi = l_crmsd_poi + [taylor_mod_poi['crmlsd'][1] + np.random.uniform(0,0.10,1)[0]]
        l_crmsd_nb = l_crmsd_nb + [taylor_mod_nb['crmlsd'][1] + np.random.uniform(0,0.10,1)[0]]
        l_crmsd_zip = l_crmsd_zip + [taylor_mod_zip['crmlsd'][1] + np.random.uniform(0,0.10,1)[0]]
        l_crmsd_zinb = l_crmsd_zinb + [taylor_mod_zinb['crmlsd'][1] + np.random.uniform(0,0.10,1)[0]]
        l_crmsd_rf = l_crmsd_rf + [taylor_mod_rf['crmlsd'][1] + np.random.uniform(0,0.10,1)[0]]

        l_ccoef_sp_poi = l_ccoef_sp_poi + [taylor_mod_poi['ccoef_sp'][0] + np.random.uniform(0,0.10,1)[0]]
        l_ccoef_sp_nb = l_ccoef_sp_nb + [taylor_mod_nb['ccoef_sp'][0] + np.random.uniform(0,0.10,1)[0]]
        l_ccoef_sp_zip = l_ccoef_sp_zip + [taylor_mod_zip['ccoef_sp'][0] + np.random.uniform(0,0.10,1)[0]]
        l_ccoef_sp_zinb = l_ccoef_sp_zinb + [taylor_mod_zinb['ccoef_sp'][0] + np.random.uniform(0,0.10,1)[0]]
        l_ccoef_sp_rf = l_ccoef_sp_rf + [taylor_mod_rf['ccoef_sp'][0] + np.random.uniform(0,0.10,1)[0]]

        l_ccoef_ke_poi = l_ccoef_ke_poi + [taylor_mod_poi['ccoef_ke'][0] + np.random.uniform(0,0.10,1)[0]]
        l_ccoef_ke_nb = l_ccoef_ke_nb + [taylor_mod_nb['ccoef_ke'][0] + np.random.uniform(0,0.10,1)[0]]
        l_ccoef_ke_zip = l_ccoef_ke_zip + [taylor_mod_zip['ccoef_ke'][0] + np.random.uniform(0,0.10,1)[0]]
        l_ccoef_ke_zinb = l_ccoef_ke_zinb + [taylor_mod_zinb['ccoef_ke'][0] + np.random.uniform(0,0.10,1)[0]]
        l_ccoef_ke_rf = l_ccoef_ke_rf + [taylor_mod_rf['ccoef_ke'][0] + np.random.uniform(0,0.10,1)[0]]

        l_names_poi = l_names_poi + [basename.format(samples_per_leaf[i], n_estimators[j])]
        l_names_nb = l_names_nb + [basename.format(samples_per_leaf[i], n_estimators[j])]
        l_names_zip = l_names_zip + [basename.format(samples_per_leaf[i], n_estimators[j])]
        l_names_zinb = l_names_zinb + [basename.format(samples_per_leaf[i], n_estimators[j])]
        l_names_rf = l_names_rf + [basename.format(samples_per_leaf[i], n_estimators[j])]


stdev_poi = np.array(l_stdev_poi)
stdev_nb = np.array(l_stdev_nb)
stdev_zip = np.array(l_stdev_zip)
stdev_zinb = np.array(l_stdev_zinb)
stdev_rf = np.array(l_stdev_rf)

crmsd_poi = np.array(l_crmsd_poi)
crmsd_nb = np.array(l_crmsd_nb)
crmsd_zip = np.array(l_crmsd_zip)
crmsd_zinb = np.array(l_crmsd_zinb)
crmsd_rf = np.array(l_crmsd_rf)

ccoef_sp_poi = np.array(l_ccoef_sp_poi)
ccoef_sp_nb = np.array(l_ccoef_sp_nb)
ccoef_sp_zip = np.array(l_ccoef_sp_zip)
ccoef_sp_zinb = np.array(l_ccoef_sp_zinb)
ccoef_sp_rf = np.array(l_ccoef_sp_rf)

ccoef_ke_poi = np.array(l_ccoef_ke_poi)
ccoef_ke_nb = np.array(l_ccoef_ke_nb)
ccoef_ke_zip = np.array(l_ccoef_ke_zip)
ccoef_ke_zinb = np.array(l_ccoef_ke_zinb)
ccoef_ke_rf = np.array(l_ccoef_ke_rf)

mksize = 14

plt.subplot(1, 2, 1)
# poidi = sm.taylor_diagram(stdev_poi, crmsd_poi, ccoef_sp_poi, markerSize=mksize, markerLabel=l_names_poi, markerLegend = 'on', colCOR='k', colSTD='k', colRMS='darkmagenta', axismax = 8.0, titleCOR="on", markerobs = 'H', markercolor="y")
# nbdi = sm.taylor_diagram(stdev_nb, crmsd_nb, ccoef_sp_nb, overlay = 'on', markerSize=mksize, markerLabel=l_names_nb, markerLegend = 'on', markercolor="r")

nbdi = sm.taylor_diagram(stdev_nb, crmsd_nb, ccoef_sp_nb, markerSize=mksize, markerLabel=l_names_nb, markerLegend = 'on', colCOR='k', colSTD='k', colRMS='darkmagenta', axismax = 8.0, titleCOR="on", markerobs = 'H', markercolor="r")
poidi = sm.taylor_diagram(stdev_poi, crmsd_poi, ccoef_sp_poi, overlay = 'on', markerSize=mksize, markerLabel=l_names_poi, markerLegend = 'on', markercolor="b")
zipdi = sm.taylor_diagram(stdev_zip, crmsd_zip, ccoef_sp_zip, overlay = 'on', markerSize=mksize, markerLabel=l_names_zip, markerLegend = 'on', markercolor="y")
zinbdi = sm.taylor_diagram(stdev_zinb, crmsd_zinb, ccoef_sp_zinb, overlay = 'on', markerSize=mksize, markerLabel=l_names_zinb, markerLegend = 'on', markercolor="g")
rfdi = sm.taylor_diagram(stdev_rf, crmsd_rf, ccoef_sp_rf, overlay = 'on', markerSize=mksize, markerLabel=l_names_rf, markerLegend = 'on', markercolor="k")

plt.subplot(1, 2, 2)
# # poidi = sm.taylor_diagram(stdev_poi, crmsd_poi, ccoef_ke_poi, markerSize=mksize, markerLabel=l_names_poi, markerLegend = 'on', colCOR='k', colSTD='k', colRMS='darkmagenta', markerobs = 'H', markercolor="y")
# nbdi = sm.taylor_diagram(stdev_nb, crmsd_nb, ccoef_ke_nb, overlay = 'on', markerSize=mksize, markerLabel=l_names_nb, markerLegend = 'on', markercolor="r")

nbdi = sm.taylor_diagram(stdev_nb, crmsd_nb, ccoef_ke_nb, markerSize=mksize, markerLabel=l_names_nb, markerLegend = 'on', colCOR='k', colSTD='k', colRMS='darkmagenta', axismax = 8.0, titleCOR="on", markerobs = 'H', markercolor="r")
poidi = sm.taylor_diagram(stdev_poi, crmsd_poi, ccoef_ke_poi, overlay = 'on', markerSize=mksize, markerLabel=l_names_poi, markerLegend = 'on', markercolor="b")
zipdi = sm.taylor_diagram(stdev_zip, crmsd_zip, ccoef_ke_zip, overlay = 'on', markerSize=mksize, markerLabel=l_names_zip, markerLegend = 'on', markercolor="y")
zinbdi = sm.taylor_diagram(stdev_zinb, crmsd_zinb, ccoef_ke_zinb, overlay = 'on', markerSize=mksize, markerLabel=l_names_zinb, markerLegend = 'on', markercolor="g")
rfdi = sm.taylor_diagram(stdev_rf, crmsd_rf, ccoef_ke_rf, overlay = 'on', markerSize=mksize, markerLabel=l_names_rf, markerLegend = 'on', markercolor="k")

# path_fig_out = r"/home/irene/Pictures/0903_Taylor_Diagrams_SpeKen_ok5.png"
# manager = plt.get_current_fig_manager()
# print(manager)
# manager.window.showMaximized()
#
# plt.pause(20)
# plt.gcf().savefig(path_fig_out, format='png', dpi=300)

plt.show()

