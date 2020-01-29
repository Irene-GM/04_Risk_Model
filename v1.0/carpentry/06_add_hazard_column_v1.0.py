import csv
import numpy as np
import gdal
import matplotlib.pyplot as plt

def getHazardInPosition(tifm, tifs, x, y):
    geotransform = tifm.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]

    xoffset = int((x - originX) / pixel_width)
    yoffset = int((y - originY) / pixel_height)

    vm = round(tifm.ReadAsArray(xoffset, yoffset, 1, 1)[0][0], 2)
    vs = round(tifs.ReadAsArray(xoffset, yoffset, 1, 1)[0][0], 2)

    if np.isnan(vm):
        vm = -1
    if np.isnan(vs):
        vs = -1

    return [vm, vs]


################
# Main program #
################

# This adds the hazard column from a single tif file, but now we need to obtain the
# maximum of the entire time-series per pixel, thus, i need to modify this program
# to eat a multiband tif and find the maximum

# These are old attempts that I am not removing just in case
# path_haz_mean = r"D:\GeoData\workspaceimg\Special\IJHG\tif\mean_std\2014_AQT_Mean_v9.tif"
# path_haz_std = r"D:\GeoData\workspaceimg\Special\IJHG\tif\mean_std\2014_AQT_Stdev_v9.tif"
# path_feat = r"M:\Documents\workspace\Special\IJHG\data\versions\country_features\v11\nl_features_v1.11_target_1km.csv"
# path_out = r"M:\Documents\workspace\Special\IJHG\data\versions\country_features\v11\nl_features_v1.11_exp_haz_target_1km.csv"
# path_exp_feat = r"M:\Documents\workspace\Special\IJHG\data\versions\country_features\v12\nl_centroids_v1.12.csv"
# path_rsk_feat = r"M:\Documents\workspace\Special\IJHG\data\versions\country_features\v12\nl_risk_features_v1.12.csv"


path_haz_mean_in = r"/media/irene/DATA/GeoData/workspaceimg/Special/IJHG/tif/hazard_mean_stdev_multiband/NL_Hazard_Mean_2006-2016_Max.tif"
path_haz_std_in = r"/media/irene/DATA/GeoData/workspaceimg/Special/IJHG/tif/hazard_mean_stdev_multiband/NL_Hazard_Stdev_2006-2016_Max.tif"

path_exp_feat_in = r"/media/irene/MyPassport/acies/workspace/Special/IJHG/data/versions/country_features/v12/nl_centroids_v1.12_nohaz.csv"
path_rsk_feat_out = r"/media/irene/MyPassport/acies/workspace/Special/IJHG/data/versions/country_features/v12/nl_centroids_v1.12.csv"

tifm = gdal.Open(path_haz_mean_in)
tifs = gdal.Open(path_haz_std_in)

band = tifs.GetRasterBand(1).ReadAsArray()
# plt.imshow(band, interpolation="None")
# plt.show()

l = []
with open(path_exp_feat_in, "r", newline="") as r:
    with open(path_rsk_feat_out, "w", newline="") as w:
        reader = csv.reader(r, delimiter=";")
        writer = csv.writer(w, delimiter=";")
        headers_exp = next(reader)
        headers = headers_exp + ["maxmeanhaz", "maxstdhaz"]
        print(headers)
        writer.writerow(headers)
        for row in reader:
            x = float(row[1])
            y = float(row[2])
            h = getHazardInPosition(tifm, tifs, x, y)
            newrow = row + h
            floatified = [float(item) for item in newrow]
            writer.writerow(newrow)
            print(floatified)
