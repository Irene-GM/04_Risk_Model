import numpy as np
import skill_metrics as sm
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


################
# Main program #
################

path_in = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/prediction_nl_four_models.csv"
path_true = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/prediction_nl_true_values.csv"

meta_m = np.loadtxt(path_in, delimiter=";", usecols=range(0, 3))
m = np.loadtxt(path_in, delimiter=";", usecols=range(3, 8))
t = np.loadtxt(path_true, delimiter=";", usecols=(3,))

labels = ['Non-Dimensional Observation', 'POI', 'NB', 'ZIP', "ZINB", "RF"]

pred_poi = m[:,0]
pred_nb = m[:,1]
pred_zip = m[:,2]
pred_zinb = m[:,3]
pred_rf = m[:,4]

target_mod_poi = sm.target_statistics(pred_poi, t, 'data')
target_mod_nb = sm.target_statistics(pred_nb, t, 'data')
target_mod_zip = sm.target_statistics(pred_zip, t, 'data')
target_mod_zinb = sm.target_statistics(pred_zinb, t, 'data')
target_mod_rf = sm.target_statistics(pred_rf, t, 'data')

taylor_mod_poi = sm.taylor_statistics(pred_poi, t, 'data')
taylor_mod_nb = sm.taylor_statistics(pred_nb, t, 'data')
taylor_mod_zip = sm.taylor_statistics(pred_zip, t, 'data')
taylor_mod_zinb = sm.taylor_statistics(pred_zinb, t, 'data')
taylor_mod_rf = sm.taylor_statistics(pred_rf, t, 'data')

target_bias = np.array([target_mod_poi['bias'], target_mod_nb['bias'], target_mod_zip['bias'], target_mod_zinb["bias"], target_mod_rf["bias"]])
target_crmsd = np.array([target_mod_poi['crmsd'], target_mod_nb['crmsd'], target_mod_zip['crmsd'], target_mod_zinb["crmsd"], target_mod_rf["crmsd"]])
target_rmsd = np.array([target_mod_poi['rmsd'], target_mod_nb['rmsd'], target_mod_zip['rmsd'], target_mod_zinb["rmsd"], target_mod_rf["rmsd"]])

taylor_stdev = np.array([taylor_mod_poi['sdev'][0], taylor_mod_poi['sdev'][1], taylor_mod_nb['sdev'][1], taylor_mod_zip['sdev'][1], taylor_mod_zinb['sdev'][1], taylor_mod_rf['sdev'][1]])
taylor_crmsd = np.array([taylor_mod_poi['crmsd'][0], taylor_mod_poi['crmsd'][1], taylor_mod_nb['crmsd'][1], taylor_mod_zip['crmsd'][1], taylor_mod_zinb['crmsd'][1], taylor_mod_rf['crmsd'][1]])
taylor_ccoef = np.array([taylor_mod_poi['ccoef'][0], taylor_mod_poi['ccoef'][1], taylor_mod_nb['ccoef'][1], taylor_mod_zip['ccoef'][1], taylor_mod_zinb['ccoef'][1], taylor_mod_rf['ccoef'][1]])

print("este print: ", taylor_mod_poi['ccoef'][0])

# sm.target_diagram(target_bias, target_crmsd, target_rmsd)

# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (15, 5),
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large'}


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          "font.size": 24,
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

pylab.rcParams.update(params)


# plt.rcParams.update({'font.size': 26})
print("This is the shape: ", taylor_stdev.shape)
print("Len of labels: ", len(labels))
sm.taylor_diagram(taylor_stdev, taylor_crmsd, taylor_ccoef, markerSize=18, markerColor="indigo", markerLabel=labels, markerLabelColor='black', colCOR='k', colSTD='k', colRMS='darkmagenta', )
# plt.xlabel('xlabel', fontsize=50)
# plt.ylabel('ylabel', fontsize='medium')

# sm.target_diagram(target_bias,target_crmsd, target_rmsd, markerLabel = labels[1:])

plt.show()