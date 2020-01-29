import gdal
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import rcParams




def correct_extent(r, shift_rows, shift_cols):
    canvas = np.empty((350, 300))
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            canvas[i, j] = r[i,j]
    return np.roll(np.roll(canvas, shift=shift_rows, axis=0), shift=shift_cols, axis=1)

def cut_extent(r):
    return r[1:351, 5:305]

def find_ris_in_exp(band, pack):
    k = 1
    dic = defaultdict(list)
    for positions in pack:
        x = positions[0]
        y = positions[1]
        l = []
        for i in range(len(x)):
            px = x[i]
            py = y[i]
            if band[px, py] < 0:
                l.append(0)
            else:
                l.append(band[px, py])
        dic[k] = np.array(l)
        k += 1
    return dic



################
# Main program #
################

path_in_exp = r"/home/irene/PycharmProjects/04_Risk_Model/data/tifs/Exposure_RD_New_vIGM_classified_4CASESWITHRH_4326.tif"
path_in_ris = r"/home/irene/PycharmProjects/04_Risk_Model/data/tifs/Risk_TickBites.tif"
path_in_haz = r"/home/irene/PycharmProjects/04_Risk_Model/data/tifs/Hazard.tif"
path_in_poi = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_Poi_1000x5_v2.tif"
path_in_nb = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_NB_1000x5_v2.tif"
path_in_zip = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_ZIP_1000x5_v2.tif"
path_in_zinb = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_ZINB_1000x5_v2.tif"

path_mask = r"/home/irene/PycharmProjects/04_Risk_Model/data/mask/mask_LGN.csv"
mask = np.loadtxt(path_mask, delimiter=";")

labels = ["Low \nexposure", "Medium \nexposure", "High \nexposure", "TB outside \nforests", "Forests \nlow recreational \nintensity", "Zero TB \nreported"]

tif_exp = gdal.Open(path_in_exp)
tif_ris = gdal.Open(path_in_ris)
tif_haz = gdal.Open(path_in_haz)

tif_poi = gdal.Open(path_in_poi)
tif_nb = gdal.Open(path_in_nb)
tif_zip = gdal.Open(path_in_zip)
tif_zinb = gdal.Open(path_in_zinb)

ban_exp = tif_exp.GetRasterBand(1).ReadAsArray()
ban_ris = tif_ris.GetRasterBand(1).ReadAsArray()
ban_haz = tif_haz.GetRasterBand(1).ReadAsArray()

ban_poi = tif_poi.GetRasterBand(1).ReadAsArray()
ban_nb = tif_nb.GetRasterBand(1).ReadAsArray()
ban_zip = tif_zip.GetRasterBand(1).ReadAsArray()
ban_zinb = tif_zinb.GetRasterBand(1).ReadAsArray()

print(ban_poi.shape, ban_exp.shape)

canvas_ris = correct_extent(ban_ris, 18, 14)
canvas_haz = cut_extent(ban_haz)
canvas_exp = cut_extent(ban_exp)

for i in range(canvas_exp.shape[0]):
    for j in range(canvas_exp.shape[1]):
        if mask[i, j] == 0:
            canvas_exp[i, j] = 8

exp_poi = np.divide(ban_poi, canvas_haz)
exp_nb = np.divide(ban_nb, canvas_haz)
exp_zip = np.divide(ban_zip, canvas_haz)
exp_zinb = np.divide(ban_zinb, canvas_haz)

plt.subplot(2, 2, 1)
plt.imshow(exp_poi, interpolation="None")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(exp_nb, interpolation="None")
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(exp_zip, interpolation="None")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(exp_zinb, interpolation="None")
plt.colorbar()
plt.show()



