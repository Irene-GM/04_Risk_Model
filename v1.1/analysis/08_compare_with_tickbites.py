import gdal
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import rcParams
from scipy.stats import skew, spearmanr, kendalltau, rankdata

def correct_extent(r, shift_rows, shift_cols):
    canvas = np.empty((350, 300))
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            canvas[i, j] = r[i,j]
    return np.roll(np.roll(canvas, shift=shift_rows, axis=0), shift=shift_cols, axis=1)

def cut_extent(r):
    return r[1:351, 6:305]

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
path_in_poi = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_Poi_1000x5.tif"
path_in_nb = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_NB_1000x5.tif"
path_in_zip = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_ZIP_1000x5.tif"
path_in_zinb = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_ZINB_1000x5.tif"

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
canvas_ris[canvas_ris>353] = -1

print(np.unique(canvas_ris, return_counts=True))

canvas_haz = cut_extent(ban_haz)
canvas_exp = cut_extent(ban_exp)

for i in range(canvas_exp.shape[0]):
    for j in range(canvas_exp.shape[1]):
        if mask[i, j] == 0:
            canvas_exp[i, j] = 8

# plt.subplot(2, 2, 1)
# plt.imshow(ban_poi, interpolation="None")
# plt.colorbar()
#
# plt.subplot(2, 2, 2)
# plt.imshow(canvas_exp, interpolation="None")
# plt.colorbar()
#
# plt.subplot(2, 2, 3)
# plt.imshow(canvas_ris, interpolation="None")
# plt.colorbar()
#
# plt.subplot(2, 2, 4)
# plt.imshow(canvas_haz, interpolation="None")
# plt.colorbar()
# plt.show()

# print(np.unique(canvas_haz, return_counts=True))

# This was to compare hazard with risk predicted, but there is nothing here
# pos_pz = np.where(canvas_haz > 0)
#
# lh, lr = [[] for i in range(2)]
# for i in range(len(pos_pz[0])):
#     px = pos_pz[0][i]
#     py = pos_pz[1][i]
#     lh.append(canvas_haz[px, py])
#     lr.append(ban_poi[px, py])
#
# lr_ran = rankdata(sorted(lh))
# lh_ran = rankdata(sorted(lr))
#
# print("Spearman: ", spearmanr(lr, lh))
# print("Kendall: ", kendalltau(lr, lh))
####################################################

pos_pz = np.where(canvas_ris > 0)
lp, lr = [[] for i in range(2)]
for i in range(len(pos_pz[0])):
    px = pos_pz[0][i]
    py = pos_pz[1][i]
    lr.append(canvas_ris[px, py])
    lp.append(ban_poi[px, py])

plt.scatter(lr, lp)
plt.show()

# pos_one = np.where(canvas_exp == 1)
# pos_two = np.where(canvas_exp == 2)
# pos_thr = np.where(canvas_exp == 3)
# pos_fou = np.where(canvas_exp == 4)
# pos_fiv = np.where(canvas_exp == 5)
# pos_six = np.where(canvas_exp == 6)
#
# pack = [pos_one, pos_two, pos_thr, pos_fou, pos_fiv, pos_six]
#
# dic_poi = find_ris_in_exp(ban_poi, pack)
# dic_nb = find_ris_in_exp(ban_nb, pack)
# dic_zip = find_ris_in_exp(ban_zip, pack)
# dic_zinb = find_ris_in_exp(ban_zinb, pack)
#
# pack_dic = [dic_poi, dic_nb, dic_zip, dic_zinb]
#
# i = 0
# all_mods = []
# for dic in pack_dic:
#     all_vals = []
#     for key in sorted(dic.keys()):
#         print(key, len(dic[key]))
#         all_vals.append(dic[key])
#     all_mods.append(all_vals)
#
#
# all_vals_poi, all_vals_nb, all_vals_zip, all_vals_zinb = [item for item in all_mods]
#
# s = 24
# labelsize = 16
# bgclr = "#DDDDDD"
# rcParams['xtick.labelsize'] = labelsize
# rcParams['ytick.labelsize'] = labelsize
#
# # plt.subplots_adjust(wspace=0.5, hspace=0.9)
# plt.tight_layout()
# fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False)
#
# ax[0,0].grid()
# bp_poi = ax[0,0].boxplot(all_vals_poi, patch_artist=True, labels=labels, medianprops=dict(color="gray"))
# ax[0,0].set_ylabel('Risk', size=20)
# # ax[0,0].set_xlabel('Types of human exposure to tick bites', size=20)
# ax[0,0].set_ylim(0, 35)
# ax[0,0].set_title("(a) TB risk: Poisson", size=s)
# ax[0,0].set_facecolor(bgclr)
#
# ax[0,1].grid()
# bp_nb = ax[0,1].boxplot(all_vals_nb, patch_artist=True, labels=labels, medianprops=dict(color="gray"))
# # ax[0,1].set_ylabel('Risk', size=20)
# # ax[0,1].set_xlabel('Types of human exposure to tick bites', size=20)
# ax[0,1].set_ylim(0, 35)
# ax[0,1].set_title("(b) TB risk: NB", size=s)
# ax[0,1].set_facecolor(bgclr)
#
# ax[1,0].grid()
# bp_zip = ax[1,0].boxplot(all_vals_zip, patch_artist=True, labels=labels, medianprops=dict(color="gray"))
# ax[1,0].set_ylabel('Risk', size=20)
# ax[1,0].set_xlabel('Types of human exposure to tick bites', size=20)
# ax[1,0].set_ylim(0, 35)
# ax[1,0].set_title("(c) TB risk: ZIP", size=s)
# ax[1,0].set_facecolor(bgclr)
#
# ax[1,1].grid()
# bp_zinb = ax[1,1].boxplot(all_vals_zinb, patch_artist=True, labels=labels, medianprops=dict(color="gray"))
# # ax[1,1].set_ylabel('Risk', size=20)
# ax[1,1].set_xlabel('Types of human exposure to tick bites', size=20)
# ax[1,1].set_ylim(0, 35)
# ax[1,1].set_title("(d) TB risk: ZINB", size=s)
# ax[1,1].set_facecolor(bgclr)
#
# # fill with colors
# colors = ['#0C3647', '#F6AC33', '#7F0E0C', "#FAE912", "#1E5932", "#0E0F0F"]
# for bplot in (bp_poi, bp_nb, bp_zip, bp_zinb):
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)
#
#
# plt.show()






