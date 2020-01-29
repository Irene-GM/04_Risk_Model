import numpy as np
import matplotlib.pyplot as plt

path_pred = r'D:\UTwente\PycharmProjects\04_Risk_Model\data\skewed_leaves\prediction_nl_four_models_v3_20T_200S.csv'
path_true = r"D:\UTwente\PycharmProjects\04_Risk_Model\data\skewed_leaves\prediction_nl_true_values.csv"

m = np.loadtxt(path_pred, delimiter=";", dtype=np.float)
p = np.loadtxt(path_true, delimiter=";", dtype=np.float)

pred_poi = m[:,3]
pred_nb = m[:,4]
pred_zip = m[:,5]
pred_zinb = m[:,6]
pred_rf = m[:,7]

true_val = p[:, 3]

plt.subplot(2, 5, 1)
plt.title("POI", size=18)
plt.plot(pred_poi, pred_poi, "-")
plt.plot(pred_poi, true_val, "o", color="gray")

plt.subplot(2, 5, 2)
plt.title("NB", size=18)
plt.plot(pred_nb, pred_nb, "-")
plt.plot(pred_nb, true_val, "o", color="gray")

plt.subplot(2, 5, 3)
plt.title("ZIP", size=18)
plt.plot(pred_zip, pred_zip, "-")
plt.plot(pred_zip, true_val, "o", color="gray")

plt.subplot(2, 5, 4)
plt.title("ZINB", size=18)
plt.plot(pred_zinb, pred_zinb, "-")
plt.plot(pred_zinb, true_val, "o", color="gray")

plt.subplot(2, 5, 5)
plt.title("RF", size=18)
plt.plot(pred_poi, pred_poi, "-")
plt.plot(pred_rf, true_val, "o", color="gray")

###########################################################

pred_poi = np.log(pred_poi+1)
pred_nb = np.log(pred_nb+1)
pred_zip = np.log(pred_zip+1)
pred_zinb = np.log(pred_zinb+1)
pred_rf = np.log(pred_rf+1)
true_val = np.log(true_val+1)

plt.subplot(2, 5, 6)
plt.title("POI", size=18)
plt.plot(pred_poi, pred_poi, "-")
plt.plot(pred_poi, true_val, "o", color="gray")

plt.subplot(2, 5, 7)
plt.title("NB", size=18)
plt.plot(pred_nb, pred_nb, "-")
plt.plot(pred_nb, true_val, "o", color="gray")

plt.subplot(2, 5, 8)
plt.title("ZIP", size=18)
plt.plot(pred_zip, pred_zip, "-")
plt.plot(pred_zip, true_val, "o", color="gray")

plt.subplot(2, 5, 9)
plt.title("ZINB", size=18)
plt.plot(pred_zinb, pred_zinb, "-")
plt.plot(pred_zinb, true_val, "o", color="gray")

plt.subplot(2, 5, 10)
plt.title("RF", size=18)
plt.plot(pred_poi, pred_poi, "-")
plt.plot(pred_rf, true_val, "o", color="gray")


plt.show()
