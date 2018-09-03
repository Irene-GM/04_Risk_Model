from osgeo import gdal, ogr
import numpy as np
import csv

def ask_for_exposure_class(rowpos, tif):
    x = float(rowpos[1])
    y = float(rowpos[2])

    geotransform = tif.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]

    xoffset = int((x - originX) / pixel_width)
    yoffset = int((y - originY) / pixel_height)

    band = tif.GetRasterBand(1)
    data = band.ReadAsArray(xoffset, yoffset, 1, 1)
    return data[0][0]


################
# Main program #
################

path_in = r"D:\UTwente\workspace\Special\IJHG\data\versions\country_features\v12\nl_risk_features_v1.12.csv"
path_exp = r"C:\Users\irene\Downloads\Exposure_RD_New_vIGM_classified_4CASESWITHRH.tif"
path_out = r"D:\UTwente\workspace\Special\IJHG\data\versions\country_features\v12\nl_risk_features_v1.12b.csv"
tifexp = gdal.Open(path_exp)

with open(path_in, "r", newline="") as r:
    with open(path_out, "w", newline="") as w:
        reader = csv.reader(r, delimiter=";")
        header = next(reader)
        writer = csv.writer(w, delimiter=";")
        writer.writerow(header+["exp"])
        for row in reader:
            exposure = ask_for_exposure_class(row, tifexp)
            newrow = row + [exposure]
            writer.writerow(newrow)