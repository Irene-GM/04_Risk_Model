import rasterio
import numpy as np
from rasterio.mask import mask
import geopandas as gpd
from osgeo import gdal, ogr, osr
import matplotlib.pyplot as plt
from shapely.geometry import mapping
from scipy import stats
import seaborn as sb

def zonal_stats(arr):
    mn = np.amin(arr)
    mx = np.amax(arr)
    md = stats.mode(arr.ravel()).mode[0]
    me = np.mean(arr)
    return [mn, mx, md, me]


################
# Main program #
################

path_lu = r"/media/irene/MyPassport/GeoData/BBG_reclassified/{0}.shp"
path_tb = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_Poi_1000x5.tif"

driver = ogr.GetDriverByName('ESRI Shapefile')

tif = gdal.Open(path_tb)

land_uses = ["Forest", "Recreation", "Agriculture", "Built-up", "DryTerrain", "WetTerrain", "transportation", "Water"]

pack = []

lnan = [np.nan for i in range(2000)]

for lu in land_uses:

    print("Processing: ", lu)

    ds = driver.Open(path_lu.format(lu), 0) # 0 means read-only. 1 means writeable.
    layer = ds.GetLayer()

    shapefile = gpd.read_file(path_lu.format(lu))
    geoms = shapefile.geometry.values # list of shapely geometries

    lmn, lmx, lmd, lme = [[] for i in range(4)]

    with rasterio.open(path_tb) as src:
        band = src.read(1)
        for geometry in geoms:
            # if counter == 0:
            #     break
            # else:
            #     counter -= 1
            geo = [mapping(geometry)]
            out_image, out_transform = mask(src, geo, crop=True)
            # plt.imshow(out_image[0,:,:], interpolation="None")
            # plt.show()
            if np.isnan(out_image.ravel()).all():
                # If we reach this point, it means that the polygon is much
                # smaller than the pixel size. Thus, all vertices of the polygon
                # lay within the same pixel. We pick the first coordinate and
                # check the value of the pixel in that location
                x, y = geo[0]["coordinates"][0][0]
                row, col = src.index(x, y)
                mn, mx, md, me = [band[row, col] for i in range(4)]
                # print("Its 111: ", row, col, band[row, col])
            else:
                # Some 'zonal statistics'
                masked = np.ma.masked_invalid(out_image)
                mn, mx, md, me = zonal_stats(masked)

                lmn.append(mn)
                lmx.append(mx)
                lmd.append(md)
                lme.append(me)

    pack.append(lme)

colors = ["royalblue"]

plt.subplots_adjust(wspace=0.5, hspace=0.5)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
ax.grid()
ax.boxplot(pack, patch_artist=True, vert=0)
ax.set_yticklabels(land_uses, size=16)
ax.set_ylabel('Land use', size=20)
ax.set_xlabel('Tick bites', size=20)
plt.show()