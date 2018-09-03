import os
import csv
import gdal
from collections import defaultdict
import numpy as np

def read_features(path):
    lmeta = []
    ldata = []
    with open(path, "r", newline="") as r:
        reader = csv.reader(r, delimiter=";")
        for row in reader:
            lmeta.append(row[0:6])
            ldata.append(row[6:27])
    return ldata, lmeta

def summary(dic):
    dicsummary = defaultdict(int)
    empty_cell = 0
    for key in sorted(dic.keys()):
        if len(dic[key]) == 0:
            empty_cell += 1
        else:
            dicsummary[len(dic[key])]+=1
    print("Empty cells: ", empty_cell)
    print("Cell w/ TB: ", 46838 - empty_cell)
    print("Summary: ", dicsummary)

def find_ids(lids, meta):
    lpos = []
    for rowid in lids:
        i = 1
        for metaid in meta[1:]:
            if int(rowid) == int(metaid[0]):
                lpos.append(i)
            i +=1
    return lpos

def ask_for_exposure_class(rowpos, tif):
    id = rowpos[0]
    x = rowpos[1]
    y = rowpos[2]

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

headers = "rowid;latitude;longitude;target;dbuiltup;dforest;drecreation;dbrr;dwrl;dwrn;dwrr;dcamping;dcaravan;dcross;dgolf;dheem;dhaven;dsafari;dwater;attr;dbath;lu;lc;maxmeanhaz;maxstdhaz".split(";")

# This file contains the correspondence between a tick bite and its associated pixel
path_in_link = r"D:\UTwente\workspace\Special\IJHG\00_data\link_pixel_and_tickbite\TB_reports_risk_per_pixel.csv"
dic = defaultdict(list)
with open(path_in_link, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    next(r)
    for row in reader:
        cellid = int(float(row[0]))
        biteid = int(float(row[1]))
        dic[cellid].append(biteid)
summary(dic)

# This is the characterization of each pixel
path_in_risk = r"D:\UTwente\workspace\Special\IJHG\data\versions\country_features\v12\nl_risk_features_v1.12.csv"
meta_risk = np.loadtxt(path_in_risk, delimiter=";", skiprows=1, usecols=range(0, 3))
risk = np.loadtxt(path_in_risk, delimiter=";", skiprows=1, usecols=range(3, 24))

# This file i dont really get why do we need it for, but it is used to read the features
# path_in_type = r"D:\UTwente\workspace\Special\IJHG\00_data\split_per_exposure\{0}"
path_in_type = r"D:\UTwente\workspace\Special\IJHG\00_data\splitted_obs_with_features\tb_{0}_risk_features_v1.12.csv"

# Path to output folder
path_out = r"D:\UTwente\workspace\Special\IJHG\00_data\zout\nl_{0}_risk_perpixel_features_v1.12.csv"
path_exp = r"C:\Users\irene\Downloads\Exposure_RD_New_vIGM_classified_4CASESWITHRH.tif"
tifexp = gdal.Open(path_exp)

for what in ["recr", "resi"]:
    with open(path_out.format(what), "w", newline="") as w:
        writer = csv.writer(w, delimiter=";")
        writer.writerow(headers+["exp"])
        print("Processing file: ", path_in_type.format(what))
        m, meta = read_features(path_in_type.format(what))
        i = 0
        for key in sorted(dic.keys()):
            lpos = find_ids(dic[key], meta)
            if len(lpos) > 0:
                exposure = ask_for_exposure_class(meta_risk[key], tifexp)
                newrow = meta_risk[key, :].tolist() + [len(lpos)] + risk[i, :].tolist() + [exposure]
                writer.writerow(newrow)
            if i % 1000 == 0:
                print("Processed: ", i)
            i += 1