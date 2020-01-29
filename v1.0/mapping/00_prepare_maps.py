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
from mpl_toolkits.basemap import Basemap


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

    print("RasterXSize ", tif.RasterXSize)
    print("RasterYSize ", tif.RasterYSize)
    print("Shape of xy source: ")
    print(xy_source.shape)

    # Create the projection objects for the convertion
    inproj = osr.SpatialReference()
    inproj.ImportFromEPSG(28992)

    outproj = osr.SpatialReference()
    outproj.ImportFromEPSG(28992)

    return [xmin, xmax, ymin, ymax, proj, xy_source, inproj, outproj]


def convertXY(xy_source, inproj, outproj):
    # function to convert coordinates
    shape = xy_source[0,:,:].shape
    size = xy_source[0,:,:].size

    # the ct object takes and returns pairs of x,y, not 2d grids
    # so the the grid needs to be reshaped (flattened) and back.
    ct = osr.CoordinateTransformation(inproj, outproj)
    xy_target = np.array(ct.TransformPoints(xy_source.reshape(2, size).T))

    xx = xy_target[:,0].reshape(shape)
    yy = xy_target[:,1].reshape(shape)

    return xx, yy

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
                print(tif.GetProjection())
                # warped = gdal.Warp("", tif, srcSRS="EPSG:28992", dstSRS='EPSG:4326', format="VRT", outputType=gdal.GDT_Float32)
                l.append(tif)


xmin, xmax, ymin, ymax, proj, xy_src, inproj, outproj = frameit(tif)

fig = plt.figure()



m = Basemap(projection="merc", lon_0=3.0, lat_0=52.0, resolution='h', llcrnrlon=3.0, llcrnrlat=50.0, urcrnrlon=8.0, urcrnrlat=54.0)
m.drawcountries(zorder=4)
m.drawcoastlines(zorder=5)
m.fillcontinents(color='tan',lake_color='lightblue', zorder=2)
m.drawmapboundary(fill_color='lightblue',zorder=1)

print(xy_src)

xx, yy = convertXY(xy_src, inproj, outproj)

ban = l[0].GetRasterBand(1).ReadAsArray()
ban[ban<0] = np.nan

print("Shapes: ")
print(xx.shape, yy.shape)
print(ban.shape)

im1 = m.pcolormesh(xx, yy, ban.T, transform=ccrs.epsg(28992), cmap=plt.cm.viridis, vmin=0, vmax=100)

# m.imshow(ban[::-1], interpolation="None", zorder=10)



plt.show()