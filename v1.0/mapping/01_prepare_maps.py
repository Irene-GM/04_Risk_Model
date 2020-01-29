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





################
# Main program #
################

path_tifs = r"/home/irene/PycharmProjects/04_Risk_Model/data/poisson_leaves/tif"

names = ["NL_TB_Risk_Poi", "NL_TB_Risk", "NL_TB_Risk_ZIP", "NL_TB_Risk_ZINB"]

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

# Poisson map

banpoi = l[0].GetRasterBand(1).ReadAsArray()
banpoi[banpoi<0] = np.nan

lcolor1 = ["#2F8A8A", "#F2CA04", "#D88E04", "#BF3503", "#721602"]
lcolor2 = ["#579C87", "#A1C181", "#FCCA46", "#FE7F2D", "#233D4D"]
lcolor3 = ["gray", "blue", "yellow", "orange", "red", "purple"]
lcolor4 = ["#ffffe0", "#ffd59b", "#ffa474", "#f47461", "#db4551", "#b81b34", "#8b0000"]
lcolor5 = ["#00008b", "#5d3e85", "#917177", "#c4a55a", "#ffd700"]
lcolor6 = ['#2e8b57','#9e5335','#8f373e','#7b1b4e','#4b0082']
lcolor7 = ['#191970','#4c497f','#777b85','#b2a981','#ffd700']


# the_cmap = LinearSegmentedColormap.from_list('mycmap', lcolor7)
the_cmap = plt.cm.viridis

axpoi = plt.subplot(1, 2, 1, projection=ccrs.epsg(28992))
axpoi.add_feature(cfeature.LAND.with_scale('10m'), facecolor="#D2B48C")
axpoi.add_feature(cfeature.OCEAN.with_scale('50m'))
axpoi.add_feature(cfeature.LAKES.with_scale('10m'))
axpoi.add_feature(cfeature.BORDERS.with_scale('10m'))

axpoi.imshow(banpoi, origin='upper', extent=NL_EXT_LEAN, transform=ccrs.epsg(28992), cmap=the_cmap)
axpoi.coastlines(resolution='10m', color='black')

cbmin, cbmax = [0, 100]
sm = plt.cm.ScalarMappable(cmap=the_cmap)
sm.set_clim(cbmin, cbmax)
sm._A = []
plt.colorbar(sm, ax=axpoi, orientation="horizontal", pad=.01)

# Negative Binomial map

bannb = l[2].GetRasterBand(1).ReadAsArray()
bannb[bannb<0] = np.nan

axnb = plt.subplot(1, 2, 2, projection=ccrs.epsg(28992))
axnb.add_feature(cfeature.LAND.with_scale('10m'), facecolor="#D2B48C")
axnb.add_feature(cfeature.OCEAN.with_scale('50m'))
axnb.add_feature(cfeature.LAKES.with_scale('10m'))
axnb.add_feature(cfeature.BORDERS.with_scale('10m'))

axnb.imshow(bannb, origin='upper', extent=NL_EXT_LEAN, transform=ccrs.epsg(28992), cmap=the_cmap)
axnb.coastlines(resolution='10m', color='black')

sm = plt.cm.ScalarMappable(cmap=the_cmap)
sm.set_clim(cbmin, cbmax)
sm._A = []
plt.colorbar(sm, ax=axnb, orientation="horizontal", pad=.01)




plt.show()
