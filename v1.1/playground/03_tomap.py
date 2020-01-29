import gdal
import numpy as np


def write_tif(placed_list, ns, nt):
    names = ["NL_TB_Risk_Poi_{0}x{1}_v3", "NL_TB_Risk_NB_{0}x{1}_v3", "NL_TB_Risk_ZIP_{0}x{1}_v3", "NL_TB_Risk_ZINB_{0}x{1}_v3", "NL_TB_Risk_RF_{0}x{1}_v3"]
    path_template = r"D:/UTwente/PycharmProjects/04_Risk_Model/data/skewed_leaves/tifs/runs/{0}.tif"
    tif_template = gdal.Open("F://RSData/KNMI/yearly/tmax/2014_tmax.tif")
    rows = tif_template.RasterXSize
    cols = tif_template.RasterYSize

    print(tif_template.GetProjection())

    # Get the origin coordinates for the tif file
    geotransform = tif_template.GetGeoTransform()

    i = 0
    for myarr in placed_list:
        path = path_template.format(names[i].format(ns, nt))
        outDs = tif_template.GetDriver().Create(path, rows, cols, 1, gdal.GDT_Float32)
        outBand = outDs.GetRasterBand(1)

        # write the data
        outDs.GetRasterBand(1).WriteArray(myarr)

        # flush data to disk, set the NoData value and calculate stats
        outBand.FlushCache()
        outBand.SetNoDataValue(np.nan)

        # georeference the image and set the projection
        outDs.SetGeoTransform(geotransform)
        outDs.SetProjection(tif_template.GetProjection())
        outDs = None
        outBand = None
        i += 1


def place(stack):
    tif = gdal.Open("F:/RSData/KNMI/yearly/tmax/2014_tmax.tif")

    geotransform = tif.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]

    canvas_poi, canvas_nb, canvas_zip, canvas_zinb, canvas_rf = [np.ones((350, 300))*-1 for i in range(5)]

    for row in stack:
        x = row[1]
        y = row[2]
        v_poi = row[3]
        v_nb = row[4]
        v_zip = row[5]
        v_zinb = row[6]
        v_rf = row[7]

        yoffset = int((x - originX) / pixel_width)
        xoffset = int((y - originY) / pixel_height)

        canvas_poi[xoffset, yoffset] = v_poi
        canvas_nb[xoffset, yoffset] = v_nb
        canvas_zip[xoffset, yoffset] = v_zip
        canvas_zinb[xoffset, yoffset] = v_zinb
        canvas_rf[xoffset, yoffset] = v_rf

    placed = [canvas_poi, canvas_nb, canvas_zip, canvas_zinb, canvas_rf]

    return placed



################
# Main program #
################

nt = 20
ns = 200

path_in = r"D:/UTwente/PycharmProjects/04_Risk_Model/data/skewed_leaves/prediction_nl_four_models_v3_20T_200S.csv"

stack = np.loadtxt(path_in, delimiter=";")

placed_list = place(stack)
write_tif(placed_list, ns, nt)

