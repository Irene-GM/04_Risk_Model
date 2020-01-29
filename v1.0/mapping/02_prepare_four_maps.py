import os
os.environ['GDAL_DATA'] = '/home/irene/miniconda3/envs/defaultpy34/share/gdal'
import gdal
import rasterio
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
from osgeo import gdal, ogr, osr
from matplotlib import colorbar, colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec


def frameit(tif):
    # get the edge coordinates and add half the resolution
    # to go to center coordinates
    gt = tif.GetGeoTransform()
    proj = tif.GetProjection()
    xres = gt[1]
    yres = gt[5]
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * tif.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * tif.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5

    xy_source = np.mgrid[xmin:xmax + xres:xres, ymax + yres:ymin:yres]

    # Create the projection objects for the convertion
    inproj = osr.SpatialReference()
    inproj.ImportFromEPSG(28992)

    outproj = osr.SpatialReference()
    outproj.ImportFromEPSG(28992)

    return [xmin, xmax, ymin, ymax, proj, xy_source, inproj, outproj]

def cm2inch(value):
    return value/2.54



################
# Main program #
################

path_tifs = r"/home/irene/PycharmProjects/04_Risk_Model/data/poisson_leaves/tif"

names = ["NL_TB_Risk_Poi", "NL_TB_Risk_NB", "NL_TB_Risk_ZIP", "NL_TB_Risk_ZINB"]

l = []
for root, dirs, files in os.walk(path_tifs):
    for file in files:
        if file.endswith("tif"):
            basename = file.split(".")[0]
            if basename in names:
                path = os.path.join(root, file)
                tif = gdal.Open(path)
                # warped = gdal.Warp("", tif, srcSRS="EPSG:28992", dstSRS='EPSG:4326', format="VRT", outputType=gdal.GDT_Float32)
                l.append(tif)


xmin, xmax, ymin, ymax, proj, xy_src, inproj, outproj = frameit(l[0])

buff = 10000
NL_EXT_LEAN = [xmin, xmax, ymin, ymax]
# NL_EXT_BUFF = [500, 299500, 300000, 640500]

the_cmap = plt.cm.viridis
cbmin, cbmax = [0, 100]

w = cm2inch(80)
h = cm2inch(60)

fig = plt.figure(1, figsize=(w,h))
# plt.subplots_adjust(hspace=0.3, wspace=0.3)
# plt.gca().set_aspect('equal', adjustable='datalim')
tit = ["Poisson", "NB", "ZIP", "ZINB"]
for i in range(len(l)):
    ax = plt.subplot(2, 2, i+1, projection=ccrs.epsg(28992))
    ax.set_title(tit[i], size=24)
    ax.set_aspect('auto')
    ax.set_adjustable('datalim')
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor="#D2B48C")
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'))


    ban = l[i].GetRasterBand(1).ReadAsArray()
    ban[ban < 0] = np.nan
    ban[ban>100] = 100

    ax.imshow(ban, origin='upper', extent=NL_EXT_LEAN, transform=ccrs.epsg(28992), cmap=the_cmap)
    ax.coastlines(resolution='10m', color='black')

    sm = plt.cm.ScalarMappable(cmap=the_cmap)
    sm.set_clim(cbmin, cbmax)
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.75)
    # cbar.ax.set_yticklabels(['< -1', '0', '> 1'])
    cbar.ax.tick_params(labelsize=24)




plt.show()
