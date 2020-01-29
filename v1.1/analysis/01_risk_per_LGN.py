import rasterio
import numpy as np
from rasterio.mask import mask
import geopandas as gpd
from osgeo import gdal, ogr, osr
from collections import defaultdict
import matplotlib.pyplot as plt
from shapely.geometry import mapping
from scipy import stats

# Re-class of LGN7
# 1 Grasslands
# 2 Agriculture
# 3 Deciduous
# 4 Coniferous
# 5 Water
# 6 Built-up
# 7 Transportation
# 8 Dunes
# 9 Heathlands
# 10 Others (marshes, foreign, coastal sands, peat, swamp, reed)

land_covers = ["Grasslands", "Agriculture", "Deciduous", "Coniferous", "Water", "Built-up", "Transportation", "Dunes", "Heathlands", "Others"]

mapping_lc = {  1:1, 2:2, 3:2, 4:2, 5:2, 6:2, 61:2, 62:2, 8:2, 9:2, 10:2,
                11:3, 12:4 ,16:5, 17:5, 18:6, 19:6, 20:6, 22:6, 23:6, 24:6, 25:7,
                26:10, 28:6, 30:10, 31:10, 32:8, 33:8, 34:8, 35:10, 36:9, 37:9, 38:9,
                39:10, 40:10, 41:10, 42:10, 43:10, 45:1}

path_lc = r"/media/irene/DATA/GeoData/LGN/tiff/1km/LGN7_1km_RD_New.tif"
path_tb = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_Poi_1000x5.tif"

tif_lc = gdal.Open(path_lc)
tif_tb = gdal.Open(path_tb)

ban_lc = tif_lc.GetRasterBand(1).ReadAsArray()
ban_tb = tif_tb.GetRasterBand(1).ReadAsArray()

dic = defaultdict(list)

for i in range(tif_lc.RasterYSize):
    print(i)
    for j in range(tif_lc.RasterXSize):
        val_lc = float(ban_lc[i, j])
        val_tb = float(ban_tb[i, j])

        if np.isnan(val_lc) == False:
            if val_lc >= 0:
                newkey = mapping_lc[val_lc]
                dic[newkey].append(val_tb)

l = []
for key in sorted(dic.keys()):
    vals = dic[key]
    l.append(vals)

plt.subplots_adjust(wspace=0.5, hspace=0.5)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
ax.grid()
ax.boxplot(l, patch_artist=True, vert=0)
ax.set_yticklabels(land_covers, size=16)
ax.set_ylabel('Land cover', size=20)
ax.set_xlabel('Tick bites', size=20)
plt.show()



