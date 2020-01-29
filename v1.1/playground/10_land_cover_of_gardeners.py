import csv
import numpy as np
from osgeo import gdal, ogr, osr
from collections import defaultdict
from operator import itemgetter
import matplotlib.pyplot as plt

path_lgn = r"/media/irene/MyPassport/GeoData/LGN/tiff/25m/NL_LGN7_25m.tif"

tif = gdal.Open(path_lgn)
ban = tif.GetRasterBand(1)

path_in = r"/home/irene/PycharmProjects/04_Risk_Model/data/NL_all_with_comments_RD_New_ready.csv"

l = []
with open(path_in, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    header = next(reader)
    for row in reader:
        env = row[-5]
        act = row[-4]
        y = float(row[3])
        x = float(row[4])

        if "tuin" in env:
            l.append((x, y))


geotransform = tif.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3]
pixel_width = geotransform[1]
pixel_height = geotransform[5]

counter = 1

dicv = defaultdict(int)
for coords in l:

    x = coords[0]
    y = coords[1]

    xoffset = int((x - originX) / pixel_width)
    yoffset = int((y - originY) / pixel_height)

    data = ban.ReadAsArray(xoffset, yoffset, 1, 1)[0][0]

    dicv[data] += 1
    counter += 1


dicls = {1:"Agr. gras", 2:"Mais", 3:"Aardappelen", 4:"Bieten", 5:"Granen", 6:"Overige gewassen", 8:"Glastuinbow", 9:"Boomgaarden",
          10:"Bloembollen", 11:"Loofbos", 12:"Naaldbos", 16:"Zoet water", 17:"Zout water", 18:"Bebouwing 1", 19:"Bebouwing 2",
          20:"Bos primair", 22:"Bos secundair", 23:"Gras primair", 24:"Kale grond", 25:"Hoofd/spoor wegen", 26:"Bebouwing buitengebied",
          28:"Gras secundair", 30:"Kwelders", 31:"Open Zand", 32:"Duinen lage veg", 33:"Duinen hoge veg", 34:"Duinheide",
          35:"Stuif/rivier zand", 36:"Heide", 37:"Matig heide", 38:"Sterk heide", 39:"Hoogveen", 40:"Bos hoogveengebied",
          41:"Moerasvegetatie", 42:"Rietvegetatie", 43:"Bos in moerasgebied", 45:"Natuurgraslanden", 61:"Boomkwekerijen",
          62:"Fruitwekerijen" }


lv = [(key, dicv[key]) for key in sorted(dicv.keys())]

lv_sort = list(sorted(lv, key=itemgetter(1)))

# lbl = [item[0] for item in lv_sort]
dat = [item[1] for item in lv_sort]
lbl = [dicls[item[0]] for item in lv_sort]

print(lv_sort)
print(len(lv_sort))

lin = np.linspace(0, 35, 36)

plt.subplot(1, 2, 1)
plt.grid(zorder=0)
plt.title("Land cover for the 'garden' tick bites", size=24)
plt.barh(lin, dat, tick_label=lbl)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

plt.show()