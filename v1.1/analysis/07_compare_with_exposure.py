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

path_in_exp = r"D:/UTwente/PycharmProjects/04_Risk_Model/data/tifs/Exposure_RD_New_vIGM_classified_4CASESWITHRH_4326.tif"
path_in_ris = r"D:/UTwente/PycharmProjects/04_Risk_Model/data/tifs/Risk_TickBites.tif"
path_in_haz = r"D:/UTwente/PycharmProjects/04_Risk_Model/data/tifs/Hazard.tif"
# path_in_poi = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_Poi_1000x5_v2.tif"
# path_in_nb = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_NB_1000x5_v2.tif"
# path_in_zip = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_ZIP_1000x5_v2.tif"
# path_in_zinb = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_ZINB_1000x5_v2.tif"
# path_in_rf = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_RF_1000x5_v2.tif"

path_in_poi = r"D:/UTwente/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_Poi_200x20_v3.tif"
path_in_nb = r"D:\UTwente\PycharmProjects\04_Risk_Model\data\skewed_leaves\tifs\runs\NL_TB_Risk_NB_200x20_v3.tif"
path_in_zip = r"D:\UTwente\PycharmProjects\04_Risk_Model\data\skewed_leaves\tifs\runs\NL_TB_Risk_ZIP_200x20_v3.tif"
path_in_zinb = r"D:\UTwente\PycharmProjects\04_Risk_Model\data\skewed_leaves\tifs\runs\NL_TB_Risk_ZINB_200x20_v3.tif"
path_in_rf = r"D:\UTwente\PycharmProjects\04_Risk_Model\data\skewed_leaves\tifs\runs\NL_TB_Risk_RF_200x20_v3.tif"

path_fig_out = r"C:/Users/irene/Pictures/paper4_fin/0204_Compare_Ris_Exp_5Mod.png"

path_mask = r"D:/Data/mask/mask_LGN.csv"
mask = np.loadtxt(path_mask, delimiter=";")

labels = ["Low \nexp.", "Medium \nexp.", "High \nexp.", "TB outside \nforests", "Forests \nlow recr. \nintensity", "Zero TB \nreported"]

tif_exp = gdal.Open(path_in_exp)
tif_ris = gdal.Open(path_in_ris)
tif_haz = gdal.Open(path_in_haz)
tif_poi = gdal.Open(path_in_poi)
tif_nb = gdal.Open(path_in_nb)
tif_zip = gdal.Open(path_in_zip)
tif_zinb = gdal.Open(path_in_zinb)
tif_rf = gdal.Open(path_in_rf)

ban_exp = tif_exp.GetRasterBand(1).ReadAsArray()
ban_ris = tif_ris.GetRasterBand(1).ReadAsArray()
ban_haz = tif_haz.GetRasterBand(1).ReadAsArray()

ban_poi = tif_poi.GetRasterBand(1).ReadAsArray()
ban_nb = tif_nb.GetRasterBand(1).ReadAsArray()
ban_zip = tif_zip.GetRasterBand(1).ReadAsArray()
ban_zinb = tif_zinb.GetRasterBand(1).ReadAsArray()
ban_rf = tif_rf.GetRasterBand(1).ReadAsArray()
ban_ori = ban_ris

print(ban_poi.shape, ban_exp.shape)

canvas_ris = correct_extent(ban_ris, 18, 14)
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

pos_one = np.where(canvas_exp == 1)
pos_two = np.where(canvas_exp == 2)
pos_thr = np.where(canvas_exp == 3)
pos_fou = np.where(canvas_exp == 4)
pos_fiv = np.where(canvas_exp == 5)
pos_six = np.where(canvas_exp == 6)

pack = [pos_one, pos_two, pos_thr, pos_fou, pos_fiv, pos_six]

dic_poi = find_ris_in_exp(ban_poi, pack)
dic_nb = find_ris_in_exp(ban_nb, pack)
dic_zip = find_ris_in_exp(ban_zip, pack)
dic_zinb = find_ris_in_exp(ban_zinb, pack)
dic_rf = find_ris_in_exp(ban_rf, pack)
dic_ori = find_ris_in_exp(canvas_ris, pack)

pack_dic = [dic_poi, dic_nb, dic_zip, dic_zinb, dic_rf, dic_ori]

i = 0
all_mods = []
for dic in pack_dic:
    all_vals = []
    for key in sorted(dic.keys()):
        all_vals.append(dic[key])
    all_mods.append(all_vals)


all_vals_poi, all_vals_nb, all_vals_zip, all_vals_zinb, all_vals_rf, all_vals_ori = [item for item in all_mods]

# clean = [item for item in ban_ris.ravel() if item != 65536]
#
# all_vals_poi = [clean] + all_vals_poi
# all_vals_nb = [clean] + all_vals_nb
# all_vals_zip = [clean] + all_vals_zip
# all_vals_zinb = [clean] + all_vals_zinb
# all_vals_rf = [clean] + all_vals_rf


s = 26
labelsize = 16
bgclr = "#DDDDDD"
rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize

# plt.subplots_adjust(wspace=0.5, hspace=0.9)
plt.tight_layout()
fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(30, 25))
# fig.delaxes(ax[1][2])

# ax[0,0].set_xlabel('Types of human exposure to tick bites', size=20)
ax[0,0].grid()
bp_poi = ax[0,0].boxplot(all_vals_poi, patch_artist=True, labels=labels, medianprops=dict(color="gray"))
ax[0,0].set_ylabel('Tick bites (Risk)', size=24, labelpad=25)
ax[0,0].set_ylim(0, 35)
ax[0,0].set_title("(a) Predicted risk: Poisson", size=s)
ax[0,0].set_facecolor(bgclr)

# ax[0,1].set_ylabel('Risk', size=20)
# ax[0,1].set_xlabel('Types of human exposure to tick bites', size=20)
ax[0,1].grid()
bp_nb = ax[0,1].boxplot(all_vals_nb, patch_artist=True, labels=labels, medianprops=dict(color="gray"))
# ax[0,1].set_ylabel('Tick bites (Risk)', size=20)
ax[0,1].set_ylim(0, 35)
ax[0,1].set_title("(b) Predicted risk: NB", size=s)
ax[0,1].set_facecolor(bgclr)

ax[0,2].grid()
bp_zip = ax[0,2].boxplot(all_vals_zip, patch_artist=True, labels=labels, medianprops=dict(color="gray"))
# ax[0,2].set_ylabel('Tick bites (Risk)', size=20)
# ax[0,2].set_xlabel('Types of human exposure to tick bites', size=20)
ax[0,2].set_ylim(0, 35)
ax[0,2].set_title("(c) Predicted risk: ZIP", size=s)
ax[0,2].set_facecolor(bgclr)

ax[1,0].grid()
bp_zinb = ax[1,0].boxplot(all_vals_zinb, patch_artist=True, labels=labels, medianprops=dict(color="gray"))
ax[1,0].set_ylabel('Tick bites (Risk)', size=24, labelpad=25)
# ax[1,1].set_ylabel('Risk', size=20)
# ax[1,0].set_xlabel('Types of human exposure to tick bites', size=20)
ax[1,0].set_ylim(0, 35)
ax[1,0].set_title("(d) Predicted risk: ZINB", size=s)
ax[1,0].set_facecolor(bgclr)

ax[1,1].grid()
bp_rf = ax[1,1].boxplot(all_vals_rf, patch_artist=True, labels=labels, medianprops=dict(color="gray"))
# ax[1,1].set_ylabel('Tick bites (Risk)', size=20)
# ax[1,1].set_ylabel('Risk', size=20)
ax[1,1].set_xlabel('Types of human exposure to tick bites', size=24, labelpad=25)
ax[1,1].set_ylim(0, 35)
ax[1,1].set_title("(e) Predicted risk: RF", size=s)
ax[1,1].set_facecolor(bgclr)

ax[1,2].grid()
bp_ori = ax[1,2].boxplot(all_vals_ori, patch_artist=True, labels=labels, medianprops=dict(color="gray"))
# ax[1,2].set_ylabel('Tick bites (Risk)', size=20)
# ax[1,1].set_ylabel('Risk', size=20)
# ax[1,1].set_xlabel('Types of human exposure to tick bites', size=20)
ax[1,2].set_ylim(0, 35)
ax[1,2].set_title("(f) Original NK + TR obs.", size=s)
ax[1,2].set_facecolor("white")


# fill with colors
colors = ['#0C3647', '#F6AC33', '#7F0E0C', "#1E5932", "#FAE912", "#0E0F0F"]
for bplot in (bp_poi, bp_nb, bp_zip, bp_zinb, bp_rf, bp_ori):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)


manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.savefig(path_fig_out, format='png', dpi=300)






