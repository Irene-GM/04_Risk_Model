import os
import itertools
from osgeo import gdal, osr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
from cartopy.io import shapereader
import matplotlib
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec

# Monkey patching because matplotlib 3 does
# not have an onHold method required by cartopy.
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.io.shapereader as shpreader
from geopy.geocoders import Nominatim
GeoAxes._pcolormesh_patched = Axes.pcolormesh

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
    outproj.ImportFromEPSG(3395)
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

def get_natural_features():
    coast = NaturalEarthFeature(category='physical', name="coastline", scale='10m', facecolor='blue', zorder=1)
    border = NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m', edgecolor='black', facecolor="tan", zorder=15)
    ocean = NaturalEarthFeature(category='physical', name="ocean", scale='50m', facecolor=cfeature.COLORS["water"], zorder=2)
    lakes = NaturalEarthFeature(category='physical', name='lakes', scale='10m', edgecolor='face',facecolor=cfeature.COLORS["water"], zorder=17)
    return [coast, border, ocean, lakes]

def get_NL_contour():
    shpfilename = shapereader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
    reader = shapereader.Reader(shpfilename)
    countries = reader.records()
    for country in countries:
        if country.attributes['GEOUNIT'] == 'Netherlands':
            return country.geometry

################
# Main program #
################

ns = 200
nt = 20
extent = [3, 7.4, 50.6, 53.7]
extent_zoom = [4.0, 7.2, 51.85, 52.7]
projection=ccrs.PlateCarree()

rows = range(0, 2)
cols = range(0, 3)

path_in = r"D:/UTwente/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/{0}.tif"

lbl_title = ["(a) RF-Poisson", "(b) RF-NB", "(c) RF-ZIP", "(d) RF-ZINB", "(e) RF-Canon"]

names_tmp = ["NL_TB_Risk_Poi_{0}x{1}_v3", "NL_TB_Risk_NB_{0}x{1}_v3", "NL_TB_Risk_ZIP_{0}x{1}_v3", "NL_TB_Risk_ZINB_{0}x{1}_v3", "NL_TB_Risk_RF_{0}x{1}_v3"]
names = [n.format(ns, nt) for n in names_tmp]

inproj, outproj = get_projections()
pairs = list(itertools.product(rows, cols))
coast, border, ocean, lakes = get_natural_features()
nl_contour = get_NL_contour()

mycmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "orange", "darkred", "black"])
mycmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "yellow", "orange", "purple", "indigo"])
mycmap3 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "deepskyblue", "turquoise", "yellow"])
cividis = ["#00204c", "#00204e", "#002150", "#002251", "#002353", "#002355", "#002456", "#002558", "#00265a", "#00265b", "#00275d", "#00285f", "#002861", "#002963", "#002a64", "#002a66", "#002b68", "#002c6a", "#002d6c", "#002d6d", "#002e6e", "#002e6f", "#002f6f", "#002f6f", "#00306f", "#00316f", "#00316f", "#00326e", "#00336e", "#00346e", "#00346e", "#01356e", "#06366e", "#0a376d", "#0e376d", "#12386d", "#15396d", "#17396d", "#1a3a6c", "#1c3b6c", "#1e3c6c", "#203c6c", "#223d6c", "#243e6c", "#263e6c", "#273f6c", "#29406b", "#2b416b", "#2c416b", "#2e426b", "#2f436b", "#31446b", "#32446b", "#33456b", "#35466b", "#36466b", "#37476b", "#38486b", "#3a496b", "#3b496b", "#3c4a6b", "#3d4b6b", "#3e4b6b", "#404c6b", "#414d6b", "#424e6b", "#434e6b", "#444f6b", "#45506b", "#46506b", "#47516b", "#48526b", "#49536b", "#4a536b", "#4b546b", "#4c556b", "#4d556b", "#4e566b", "#4f576c", "#50586c", "#51586c", "#52596c", "#535a6c", "#545a6c", "#555b6c", "#565c6c", "#575d6d", "#585d6d", "#595e6d", "#5a5f6d", "#5b5f6d", "#5c606d", "#5d616e", "#5e626e", "#5f626e", "#5f636e", "#60646e", "#61656f", "#62656f", "#63666f", "#64676f", "#65676f", "#666870", "#676970", "#686a70", "#686a70", "#696b71", "#6a6c71", "#6b6d71", "#6c6d72", "#6d6e72", "#6e6f72", "#6f6f72", "#6f7073", "#707173", "#717273", "#727274", "#737374", "#747475", "#757575", "#757575", "#767676", "#777776", "#787876", "#797877", "#7a7977", "#7b7a77", "#7b7b78", "#7c7b78", "#7d7c78", "#7e7d78", "#7f7e78", "#807e78", "#817f78", "#828078", "#838178", "#848178", "#858278", "#868378", "#878478", "#888578", "#898578", "#8a8678", "#8b8778", "#8c8878", "#8d8878", "#8e8978", "#8f8a78", "#908b78", "#918c78", "#928c78", "#938d78", "#948e78", "#958f78", "#968f77", "#979077", "#989177", "#999277", "#9a9377", "#9b9377", "#9c9477", "#9d9577", "#9e9676", "#9f9776", "#a09876", "#a19876", "#a29976", "#a39a75", "#a49b75", "#a59c75", "#a69c75", "#a79d75", "#a89e74", "#a99f74", "#aaa074", "#aba174", "#aca173", "#ada273", "#aea373", "#afa473", "#b0a572", "#b1a672", "#b2a672", "#b4a771", "#b5a871", "#b6a971", "#b7aa70", "#b8ab70", "#b9ab70", "#baac6f", "#bbad6f", "#bcae6e", "#bdaf6e", "#beb06e", "#bfb16d", "#c0b16d", "#c1b26c", "#c2b36c", "#c3b46c", "#c5b56b", "#c6b66b", "#c7b76a", "#c8b86a", "#c9b869", "#cab969", "#cbba68", "#ccbb68", "#cdbc67", "#cebd67", "#d0be66", "#d1bf66", "#d2c065", "#d3c065", "#d4c164", "#d5c263", "#d6c363", "#d7c462", "#d8c561", "#d9c661", "#dbc760", "#dcc860", "#ddc95f", "#deca5e", "#dfcb5d", "#e0cb5d", "#e1cc5c", "#e3cd5b", "#e4ce5b", "#e5cf5a", "#e6d059", "#e7d158", "#e8d257", "#e9d356", "#ebd456", "#ecd555", "#edd654", "#eed753", "#efd852", "#f0d951", "#f1da50", "#f3db4f", "#f4dc4e", "#f5dd4d", "#f6de4c", "#f7df4b", "#f9e049", "#fae048", "#fbe147", "#fce246", "#fde345", "#ffe443", "#ffe542", "#ffe642", "#ffe743", "#ffe844", "#ffe945"]
mycmap_cividis = matplotlib.colors.LinearSegmentedColormap.from_list("", cividis)

# Create the figure and basemap object
# fig, ax = plt.subplots(nrows=2, ncols=4)
# fig.subplots_adjust(wspace=0.1, hspace=0.1)
# fig.delaxes(ax[1][2])

gs = gridspec.GridSpec(3, 5, height_ratios=[10, 10, 4])
gs.update(wspace=0.1)
# gs.update(left=0.05, right=0.48, wspace=0.05)

ax1 = plt.subplot(gs[0, 0], projection=ccrs.Mercator())
ax2 = plt.subplot(gs[0, 1], projection=ccrs.Mercator())
ax3 = plt.subplot(gs[0, 2], projection=ccrs.Mercator())
ax4 = plt.subplot(gs[0, 3], projection=ccrs.Mercator())
ax5 = plt.subplot(gs[0, 4], projection=ccrs.Mercator())
ax6 = plt.subplot(gs[1, 0:2], projection=ccrs.Mercator())
ax7 = plt.subplot(gs[1, 3:5], projection=ccrs.Mercator())
ax8 = plt.subplot(gs[2, :],  projection=ccrs.Mercator())

laxes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]


for k in range(len(names)):
    i, j = pairs[k]
    path_tif = path_in.format(names[k])
    ds = gdal.Open(path_tif)
    xmin, xmax, ymin, ymax, xres, yres = framing_tif(ds)
    xy_source = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]
    xx, yy = convertXY(xy_source, inproj, outproj)

    # Add layers to the map
    ban_mod = ds.GetRasterBand(1).ReadAsArray()
    ban_mod[ban_mod<0] = np.nan
    ban_mod[ban_mod > 30] = 30

    ban_mod = np.log10(ban_mod + 1)
    laxes[k].set_title(lbl_title[k], size=24)
    laxes[k].set_extent(extent)
    laxes[k].add_feature(coast, edgecolor='black', linewidth=2)
    laxes[k].add_feature(border, edgecolor="black", linewidth=2)
    laxes[k].add_feature(lakes, edgecolor="black")
    laxes[k].add_feature(ocean, edgecolor="black")
    laxes[k].add_geometries(nl_contour, ccrs.PlateCarree(), facecolor="none", edgecolor="black", zorder=20)

    im = laxes[k].pcolormesh(xx, yy, ban_mod, zorder=16, cmap=plt.cm.RdYlGn_r, vmin=0, vmax=np.log10(30))



# ADD ZOOMED ZIP
path_tif = path_in.format(names[2])
ds = gdal.Open(path_tif)
xmin, xmax, ymin, ymax, xres, yres = framing_tif(ds)
xy_source = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]
xx, yy = convertXY(xy_source, inproj, outproj)
# Add layers to the map
ban_mod = ds.GetRasterBand(1).ReadAsArray()
ban_mod[ban_mod<0] = np.nan
# ban_mod[ban_mod>30] = 30
ban_mod = np.log10(ban_mod+1)

laxes[5].set_title("(f) Zoomed RF-ZIP", size=24)
laxes[5].set_extent(extent_zoom)
laxes[5].add_feature(coast, edgecolor='black', linewidth=2)
laxes[5].add_feature(border, edgecolor="black", linewidth=2)
laxes[5].add_feature(lakes, edgecolor="black")
laxes[5].add_feature(ocean, edgecolor="black")

laxes[5].add_geometries(nl_contour, ccrs.Mercator(), facecolor="none", edgecolor="black", zorder=20)
im = laxes[5].pcolormesh(xx, yy, ban_mod, zorder=16, cmap=plt.cm.RdYlGn_r, vmin=0, vmax=np.log10(30))

# ADD ZOOMED ZINB
path_tif = path_in.format(names[3])
ds = gdal.Open(path_tif)
xmin, xmax, ymin, ymax, xres, yres = framing_tif(ds)
xy_source = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]
xx, yy = convertXY(xy_source, inproj, outproj)
# Add layers to the map
ban_mod = ds.GetRasterBand(1).ReadAsArray()
ban_mod[ban_mod<0] = np.nan
# ban_mod[ban_mod>30] = 30
ban_mod = np.log10(ban_mod+1)

laxes[6].set_title("(g) Zoomed RF-ZINB", size=24)
laxes[6].set_extent(extent_zoom)
laxes[6].add_feature(coast, edgecolor='black', linewidth=2)
laxes[6].add_feature(border, edgecolor="black", linewidth=2)
laxes[6].add_feature(lakes, edgecolor="black")
laxes[6].add_feature(ocean, edgecolor="black")

laxes[6].add_geometries(nl_contour, ccrs.Mercator(), facecolor="none", edgecolor="black", zorder=20)
im = laxes[6].pcolormesh(xx, yy, ban_mod, zorder=16, cmap=plt.cm.RdYlGn_r, vmin=0, vmax=np.log10(30))



# Add colorbar to the map
cax, kw = matplotlib.colorbar.make_axes(laxes[7], location='top', shrink=0.55, pad=0.1)
cbar = plt.colorbar(im, cax=cax, **kw, )
cbar.set_label("Tick bites (log-10 scale)", size=24, labelpad=-100)
cbar.ax.tick_params(labelsize=20)
cax.set_aspect('auto')
cax.yaxis.set_ticks_position('both')


laxes[7].set_visible(False)

geolocator = Nominatim()

loc_utr = geolocator.geocode("Utrecht, NL")
loc_ams = geolocator.geocode("Amersfoort, NL")
loc_ape = geolocator.geocode("Apeldoorn, NL")
loc_ams = geolocator.geocode("Amsterdam, NL")
loc_arn = geolocator.geocode("Arnhem, NL")
loc_gou = geolocator.geocode("Gouda, NL")
loc_zwo = geolocator.geocode("Zwolle, NL")
loc_dev = geolocator.geocode("Deventer, NL")
loc_rot = geolocator.geocode("Rotterdam, NL")
loc_ens = geolocator.geocode("Enschede, NL")


pack = [loc_utr, loc_ams, loc_ape, loc_ams, loc_arn, loc_gou, loc_zwo, loc_rot, loc_ens]

lons = [item.longitude for item in pack]
lats = [item.latitude for item in pack]
city = [item.address.split(",")[0] for item in pack]


xoffset = 0.015
yoffset = 0.015

color_marker = "black"
s = 9

for i in range(len(pack)):
    laxes[5].plot(lons[i], lats[i], color=color_marker, linewidth=2, marker='s', transform=ccrs.Geodetic(), markersize=s, zorder=18)
    if city[i] == "Enschede":
        txt = laxes[5].text(lons[i] - xoffset * 20, lats[i] + yoffset, city[i], color='black', transform=ccrs.Geodetic(), size=18, variant='small-caps', fontname='Sans', zorder=20)
    else:
        txt = laxes[5].text(lons[i] + xoffset, lats[i] + yoffset, city[i], color='black', transform=ccrs.Geodetic(), size=18, variant='small-caps', fontname='Sans', zorder=20)

    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='white')])


for i in range(len(pack)):
    laxes[6].plot(lons[i], lats[i], color=color_marker, linewidth=2, marker='s', transform=ccrs.Geodetic(), markersize=s, zorder=18)
    if city[i] == "Enschede":
        txt = laxes[6].text(lons[i] - xoffset*20, lats[i] + yoffset, city[i], color='black', transform=ccrs.Geodetic(), size=18, variant='small-caps', fontname='Sans', zorder=20)
    else:
        txt = laxes[6].text(lons[i] + xoffset, lats[i] + yoffset, city[i], color='black', transform=ccrs.Geodetic(), size=18, variant='small-caps', fontname='Sans', zorder=20)

    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='white')])

path_fig_out = r"C:/Users/irene/Pictures/paper4_fin/1011_FiveMaps_GnYlRd_log10_fullext.tif"
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.pause(10)
plt.gcf().savefig(path_fig_out, format='png', dpi=300)

# plt.show()