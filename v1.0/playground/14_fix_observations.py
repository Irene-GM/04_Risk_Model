import csv
import gdal
import numpy as np
import matplotlib.pyplot as plt

path_in = r"F:/acies/workspace/Special/IJHG/data/versions/country_features/v12/nl_risk_features_v1.12.csv"
path_lgn = r"F:/season_server/PycharmProjects/NL_predictors/data/NL_LGN_1km_Loofd_Naald_Gras.tif"
path_mask = r"F:/season_server/PycharmProjects/NL_predictors/data/mask_LGN_v2.csv"
path_out = r"D:/UTwente/PycharmProjects/04_Risk_Model/data/out/NL_Tick_Bites_2006_2016_NK_TR_features_v2.csv"

tif = gdal.Open(path_lgn)
mask = tif.GetRasterBand(1).ReadAsArray()

geotransform = tif.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3]
pixel_width = geotransform[1]
pixel_height = geotransform[5]

mask[np.isnan(mask)] = -99
mask[mask < -1] = -1
print(np.unique(mask, return_counts=True))

lvals = [11, 12, 20, 22, 23, 28, 45]

with open(path_in, "r", newline="") as r:
    with open(path_out, "w", newline="") as w:
        reader = csv.reader(r, delimiter=";")
        writer = csv.writer(w, delimiter=";")
        header = next(reader)
        writer.writerow(header)
        for row in reader:
            lat = float(row[2])
            lon = float(row[1])
            yoffset = int((lon - originX) / pixel_width)
            xoffset = int((lat - originY) / pixel_height)
            if mask[xoffset, yoffset] in lvals:
                writer.writerow(row)
            else:
                if mask[xoffset, yoffset] == -1:
                    newrow = row[:-2] + [-1, -1]
                    writer.writerow(newrow)
                # else:
                    # newrow = row[:-2] + [-1, -1]
                    # writer.writerow(newrow)
