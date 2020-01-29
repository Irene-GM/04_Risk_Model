from mpl_toolkits.basemap import Basemap
import osr, gdal
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import matplotlib
import matplotlib.animation as animation
import datetime
import os

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

def get_edge_coordinates(gt, ds):
    xres = gt[1]
    yres = gt[5]
    # get the edge coordinates and add half the resolution to go to center coordinates
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5
    return [xmin, xmax, ymin, ymax, xres, yres]

def get_projections(proj, m):
    # Create the projection objects for the convertion
    inproj = osr.SpatialReference()
    inproj.ImportFromWkt(proj)

    # Get the target projection from the basemap object
    outproj = osr.SpatialReference()
    outproj.ImportFromProj4(m.proj4string)

    return [inproj, outproj]

def add_decorations(m):
    # path_rail = r"/nobackup/users/garciama/data/geodata/vector/NATREG_SpoorWegen/railways"
    # rails = m.readshapefile(path_rail, 'railways', drawbounds=True, color="#FFFFFF", zorder=6)
    # path_rail = r"/nobackup/users/garciama/data/geodata/vector/NL_ADM/2018_Imergis_provinciegrenzen_lijn-shp/2018_Imergis_provinciegrenzen_lijn_gcs"
    # rails = m.readshapefile(path_rail, '2018_Imergis_provinciegrenzen_lijn_gcs', drawbounds=True, color="#CCCCCC", zorder=6)
    # col = rails[-1]
    # col.set_linestyle('dotted')
    m.drawcountries(zorder=4)
    m.drawcoastlines(zorder=5)
    m.fillcontinents(color='tan', lake_color='lightblue', zorder=2)
    m.drawmapboundary(fill_color='lightblue', zorder=1)
    return m


def update(i, m, xx, yy, data, sta_date):
    date = sta_date + datetime.timedelta(int(i))
    # print(i, date, sta_doy + int(i))
    plt.title("{0}/{1}/{2}".format(str(date.day).zfill(2), str(date.month).zfill(2), str(date.year).zfill(2)))

    z = data[i,:,:]
    z[z<=0] = np.nan
    print(i, np.nanmin(z), np.nanmax(z))
    # print("nans? ", np.isnan(dat).any())
    # print("zero: ", dat[0,0])
    # z = np.ma.masked_array(dat, dat == -1.1)

    # names = np.array(["Not suitable", "Suitable"])
    # names = np.array(["Low suitability", "", "", "", "", "High suitability"])
    # formatter = plt.FuncFormatter(lambda val, loc: names[val])
    # cmap = colors.ListedColormap(['white', '#1A7DD1'])
    cmap = plt.cm.BrBG

    im1 = m.pcolormesh(xx, yy, z.T, cmap=cmap, vmin=0, vmax=65, zorder=3)

    cbar = m.colorbar(im1, location='bottom', pad="5%")
    cbar.set_label('Tick Activity', size=22, labelpad=20)
    # cbar.ax.set_xticklabels(names)

    # if date == datetime.datetime(2011, 12, 18):
    #     plt.show()
    return m

def read_tif_stack(path_tif, sd, ed):
    l = []
    for root, dirs, files in os.walk(path_tif):
        for file in sorted(files):
            if file.endswith(".tif"):
                # d = datetime.datetime.strptime(file.split(".")[0].split("_")[1], "%Y-%m-%d")
                spl = file.split(".")[0].split("_")
                d = datetime.datetime(int(spl[-3]), int(spl[-2]), int(spl[-1]))
                if sd <= d <= ed:
                    path_curr = os.path.join(root, file)
                    ds = gdal.Open(path_curr, gdal.GA_ReadOnly)
                    print("Projection: ", ds.GetProjection())
                    data = ds.ReadAsArray()
                    l.append(data)
    stack_raw = np.array(l)
    # stack = np.swapaxes(stack_raw, 0, 2)
    proj = ds.GetProjection()
    gt = ds.GetGeoTransform()
    return [stack_raw, ds, gt, proj]


def create_map(path_tif, sd, ed):

    stack, ds, gt, proj = read_tif_stack(path_tif, sd, ed)


    m = Basemap(projection='merc', lon_0=3.0, lat_0=52.0, resolution='h',llcrnrlon=3.0, llcrnrlat=50.0, urcrnrlon=8.0, urcrnrlat=54.0)

    m = add_decorations(m)

    inproj, outproj = get_projections(proj, m)

    xmin, xmax, ymin, ymax, xres, yres = get_edge_coordinates(gt, ds)
    xy_source = np.mgrid[xmin:xmax + xres:xres, ymax + yres:ymin:yres]

    print("\n\n")
    print("Inproj")
    print(inproj)
    print()
    print("Outproj")
    print(outproj)
    print("\n\n")

    xx, yy = convertXY(xy_source, inproj, outproj)

    # cmap = colors.ListedColormap(['white', '#1A7DD1'])

    return [m, xx, yy, stack]



################
# Main program #
################

font = {'family' : 'sans-serif',
        'size'   : 18}
matplotlib.rc('font', **font)

# writer = animation.writers['imagemagick']
writer = animation.writers['ffmpeg']
# writer = writer(fps=2, metadata=dict(artist='Me'), bitrate=2400)

h_in_inches = 12
w_in_inches = 9

fig = plt.figure(tight_layout=True, figsize=(h_in_inches, w_in_inches))
# fig.set_size_inches(h_in_inches, w_in_inches, True)

mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
mng.full_screen_toggle()

path_tif = r"G:/season_server/PycharmProjects/NL_predictors/data/versions/v10/maps_v10/2014/"

path_out_ani = r"C:/Users/irene/Videos/animationphd/ani.mp4"


sta_date = datetime.datetime(2014, 1, 1)
end_date = datetime.datetime(2015, 1, 1)

sta_doy = sta_date.timetuple().tm_yday
end_doy = end_date.timetuple().tm_yday


m, xx, yy, data = create_map(path_tif, sta_date, end_date)

print(xx)

print("Minmax")
print(np.amin(data))
print(np.amax(data))

ani = animation.FuncAnimation(plt.gcf(), update, frames=20, fargs=(m, xx, yy, data, sta_date), interval=600)

# plt.show()

path_tif_out = path_out_ani.format(sta_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

ani.save(path_out_ani, dpi=300)