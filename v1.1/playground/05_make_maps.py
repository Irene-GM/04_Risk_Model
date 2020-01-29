import os
import geopandas
from osgeo import gdal, osr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
from cartopy.io import shapereader
import matplotlib

from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

def default_proj_lib():
    proj_lib = os.getenv('PROJ_LIB')
    if proj_lib not in (None, 'PROJ_LIB'):
        return proj_lib
    try:
        import conda
    except ImportError:
        conda = None
    if conda is not None or os.getenv('CONDA_PREFIX') is None:
        conda_file_dir = conda.__file__
        conda_dir = conda_file_dir.split('lib')[0]
        proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
        if os.path.exists(proj_lib):
            return proj_lib
        return None
    return None


def framing_tif(ds):
    # get the edge coordinates and add half the resolution
    # to go to center coordinates
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    xres = gt[1]
    yres = gt[5]
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5

    ds = None

    # create a grid of xy coordinates in the original projection
    xy_source = np.mgrid[xmin:xmax + xres:xres, ymax + yres:ymin:yres]

    return [gt, proj, xres, yres, xmin, xmax, ymin, ymax, xy_source]


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

    return xx.T, yy.T

################
# Main program #
################

# ax.set_facecolor(cfeature.COLORS["water"])

extent = [3, 7.4, 50.6, 53.7]

path_in = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/NL_TB_Risk_Poi_1000x5.tif"

ds = gdal.Open(path_in)
proj = ds.GetProjection()

ban_poi = ds.GetRasterBand(1).ReadAsArray()

# xx, yy = convertXY(xy_source, inproj, outproj)
# im1 = m.pcolormesh(xx, yy, z.T, cmap=mycmap, norm=mynorm, zorder=3)

# # Create the figure and basemap object
# # fig, axes = plt.subplots(nrows=2, ncols=3)
#
projection=ccrs.Mercator()
print(projection.proj4_params)

fig, ax = plt.subplots(subplot_kw=dict(projection=projection))
#
#
# Create the projection objects for the convertion
inproj = osr.SpatialReference()
inproj.ImportFromEPSG(28992)
outproj = osr.SpatialReference()
outproj.ImportFromEPSG(3395)

gt = ds.GetGeoTransform()
proj = ds.GetProjection()
xres = gt[1]
yres = gt[5]
xmin = gt[0] + xres * 0.5
xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
ymax = gt[3] - yres * 0.5

ds = None

# create a grid of xy coordinates in the original projection
xy_source = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]

print(ban_poi.shape)

xx, yy = convertXY(xy_source, inproj, outproj)

ban_poi[ban_poi<0] = np.nan

print(cfeature.COLORS)

im1 = ax.pcolormesh(xx, yy, ban_poi, zorder=16, cmap=plt.cm.RdYlGn_r, vmin=0, vmax=35)

coast = NaturalEarthFeature(category='physical',  name="coastline", scale='10m', facecolor='blue',zorder=1)
border = NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m', edgecolor='black', facecolor="tan", zorder=15)
ocean = NaturalEarthFeature(category='physical', name="ocean", scale='50m', facecolor=cfeature.COLORS["water"], zorder=2)
lakes = NaturalEarthFeature(category='physical', name='lakes', scale='10m', edgecolor='face', facecolor=cfeature.COLORS["water"], zorder=17)

feat_coast = ax.add_feature(coast, edgecolor='black', linewidth=2)
feat_bound = ax.add_feature(border, edgecolor="black", linewidth=2)
feat_lakes = ax.add_feature(lakes, edgecolor="black")
feat_ocean = ax.add_feature(ocean)

ax.set_extent(extent)

shpfilename = shapereader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
reader = shapereader.Reader(shpfilename)
countries = reader.records()
for country in countries:
    if country.attributes['GEOUNIT'] == 'Netherlands':
        ax.add_geometries(country.geometry, ccrs.PlateCarree(), facecolor="none", edgecolor="black", zorder=20)


# resolution = '10m'
# category = 'cultural'
# name = 'admin_0_countries'
# shpfilename = shapereader.natural_earth(resolution, category, name)
#
# # read the shapefile using geopandas
# df = geopandas.read_file(shpfilename)
#
# # read the german borders
# poly = df.loc[df['GEOUNIT'] == "Netherlands"]['geometry'].values[0]
# print(poly)

# ax.add_geometries(poly, crs=ccrs.Mercator(), facecolor='blue', edgecolor='black', zorder=20)

cax, kw = matplotlib.colorbar.make_axes(ax,location='bottom',pad=0.05,shrink=0.7)
cbar=fig.colorbar(im1, cax=cax, **kw)
cbar.set_label('Tick bites', size=24, labelpad=20)
cbar.ax.tick_params(labelsize=18)


plt.show()