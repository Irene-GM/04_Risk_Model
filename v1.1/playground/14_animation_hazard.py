import os
import datetime
import matplotlib.pyplot as plt
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import seaborn as sb
import numpy as np
from osgeo import ogr, osr, gdal
import matplotlib.animation as animation


def get_natural_features():
    coast = NaturalEarthFeature(category='physical', name="coastline", scale='10m', facecolor='blue', zorder=1)
    border = NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m', edgecolor='black', facecolor="tan", zorder=15)
    ocean = NaturalEarthFeature(category='physical', name="ocean", scale='10m', facecolor=cfeature.COLORS["water"], zorder=2)
    lakes = NaturalEarthFeature(category='physical', name='lakes', scale='10m', edgecolor='face',facecolor=cfeature.COLORS["water"], zorder=17)
    return [coast, border, ocean, lakes]

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

def get_projections():
    inproj = osr.SpatialReference()
    inproj.ImportFromEPSG(28992)
    outproj = osr.SpatialReference()
    # outproj.ImportFromEPSG(3395)
    outproj.ImportFromEPSG(28992)
    return inproj, outproj

def framing_tif(ds):
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    xres = gt[1]
    yres = gt[5]
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5

    ds = None

    return [xmin, xmax, ymin, ymax, xres, yres]


# def make_figure():
#
#     proj_equivalent = ccrs.Stereographic(central_longitude=5.3876388888, central_latitude=52.15616055555,
#                                          false_easting=155000, false_northing=463000, scale_factor=0.9999079)
#
#     subplot_kw = dict(projection=proj_equivalent)
#     fig, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw=subplot_kw)
#
#     coast, border, ocean, lakes = get_natural_features()
#
#     x0, x1 = -4.2e4, +3.2e5
#     y0, y1 = 2.8e5, 6.5e5
#     ax.set_extent((x0, x1, y0, y1), crs=proj_equivalent)
#
#     ax.add_feature(coast, edgecolor='black', linewidth=2)
#     ax.add_feature(border, edgecolor="black", linewidth=2)
#     ax.add_feature(lakes, edgecolor="black")
#     ax.add_feature(ocean, edgecolor="black")
#
#     return [fig, ax]


def update(i, ax, xx, yy, data, sta_date):
    print(i, "am here")
    # date = sta_date + datetime.timedelta(int(i))
    plt.title(i)

    z = data[i,:,:]
    print("Values of z: ", np.unique(z, return_counts=True))

    cmap = plt.cm.BrBG

    ax.pcolormesh(yy, xx, z, cmap=cmap, vmin=0, vmax=1, zorder=3)
    return ax


# def read_tif_stack(path_tif, sd, ed):
#     l = []
#     for root, dirs, files in os.walk(path_tif):
#         for file in sorted(files):
#             if file.endswith(".tif"):
#                 spl = file.split(".")[0].split("_")
#
#                 # d = datetime.datetime.strptime(file.split(".")[0].split("_")[1], "%Y-%m-%d")
#                 d = datetime.datetime(int(spl[-3]), int(spl[-2]), int(spl[-1]))
#                 if sd <= d <= ed:
#                     path_curr = os.path.join(root, file)
#                     ds = gdal.Open(path_curr, gdal.GA_ReadOnly)
#                     data = ds.GetRasterBand(1).ReadAsArray()
#                     data[data<0] = np.nan
#                     l.append(data)
#
#     # no: 10, 01, 20, 02, 12
#     # algo: 12,21,
#     stack_raw = np.array(l)
#     # stack = np.swapaxes(stack_raw, 2, 1)
#     proj = ds.GetProjection()
#     gt = ds.GetGeoTransform()
#     return [stack_raw, ds, gt, proj]



################
# Main program #
################

sta_date = datetime.datetime(2014, 1, 1)
end_date = datetime.datetime(2015, 1, 1)

path_in = r"G:/season_server/PycharmProjects/NL_predictors/data/versions/v10/maps_v10/2014/"
path_ou = r"C:\Users\irene\Videos/animationphd/ani.gif"

path_sample = r"G:/season_server/PycharmProjects/NL_predictors/data/versions/v10/maps_v10/2014/NL_Map_RF_NL_Prediction_2014_03_12.tif"


# ------------------------------------------------------------------------------------------------------------#
print("Getting frame")
inproj, outproj = get_projections()
ds = gdal.Open(path_sample, gdal.GA_ReadOnly)
xmin, xmax, ymin, ymax, xres, yres = framing_tif(ds)
xy_source = np.mgrid[xmin:xmax + xres:xres, ymax + yres:ymin:yres]
xx, yy = convertXY(xy_source, inproj, outproj)


# ------------------------------------------------------------------------------------------------------------#
print("Making axes")

proj_equivalent = ccrs.Stereographic(central_longitude=5.3876388888, central_latitude=52.15616055555,
                                     false_easting=155000, false_northing=463000, scale_factor=0.9999079)

subplot_kw = dict(projection=proj_equivalent)
fig, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw=subplot_kw)

coast, border, ocean, lakes = get_natural_features()

x0, x1 = -4.2e4, +3.2e5
y0, y1 = 2.8e5, 6.5e5
ax.set_extent((x0, x1, y0, y1), crs=proj_equivalent)

ax.add_feature(coast, edgecolor='black', linewidth=2)
ax.add_feature(border, edgecolor="black", linewidth=2)
ax.add_feature(lakes, edgecolor="black")
ax.add_feature(ocean, edgecolor="black")

# ------------------------------------------------------------------------------------------------------------#
print("Reading data")
l = []
for root, dirs, files in os.walk(path_in):
    for file in sorted(files):
        if file.endswith(".tif"):
            spl = file.split(".")[0].split("_")

            # d = datetime.datetime.strptime(file.split(".")[0].split("_")[1], "%Y-%m-%d")
            d = datetime.datetime(int(spl[-3]), int(spl[-2]), int(spl[-1]))
            if sta_date <= d <= end_date:
                path_curr = os.path.join(root, file)
                ds = gdal.Open(path_curr, gdal.GA_ReadOnly)
                data = ds.GetRasterBand(1).ReadAsArray()
                data[data<0] = np.nan
                l.append(data)


stack = np.array(l)
proj = ds.GetProjection()
gt = ds.GetGeoTransform()

# ------------------------------------------------------------------------------------------------------------#
print("Animating")

def animate(i, ax, xx, yy, stack):
    print(i, xx.shape, yy.shape, stack.shape)
    # ax.clear() #Clear previous data
    # ax.pcolormesh(lon*180/np.pi,lat*180/np.pi,drm, transform=ccrs.PlateCarree(),cmap='seismic',vmin=-vlim,vmax=vlim)
    z = stack[i, :, :]
    print("Z: ", z.shape)
    print("ZT: ", z.T.shape)
    # plt.imshow(z, interpolation="None")
    # plt.show()
    ax.pcolormesh(yy, xx, z[::-1], transform=proj_equivalent)

    ax.set_title(i)
    return ax

ani = animation.FuncAnimation(fig, animate, frames=3, fargs=(ax, xx, yy, stack), interval=600)

# ani = animation.FuncAnimation(fig, animate, frames=5, fargs=(ax, xx, yy, stack, sta_date), interval=600)
# ani = animation.FuncAnimation(fig, update, frames=5)
# print("Done")

ani.save(path_ou, writer='imagemagick', dpi=300)