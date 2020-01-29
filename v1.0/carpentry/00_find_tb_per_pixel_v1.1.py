from osgeo import gdal, ogr, osr
import csv
import os

################
# Main program #
################

# v1.1:
# -------
# Initially, this program took a subset of the original tick bites.
# Now I think its better to subset once I have the full table of
# features to model the risk, because we can split between
# recreational and residential users.
#
# Filtering by "forest no garden" is a too sharp cut considering
# how messy is landscape in NL. We may have residential reports
# that occured in a nearby patch of forest, so we have to allow
# some flexibility to account for the mismatch between the positional
# accuracy and the reported activity. It doesnt seem that
# a distance-to-forest threshold really makes sense, so perhaps
# ita makes sense to try a higher-level cut: recreational and residential
# and proceed with two risk models.

path_grid = r"/home/irene/geodata/500m_clipped/500m_clipped.shp"
path_bites = r"/media/irene/DATA/GeoData/Teken_Observations2012-2016/TickBites_Dec16/NL_TickBites_Dec16_RD_New.shp"
path_out = r"/home/irene/PycharmProjects/04_Risk_Model/data/no_split/500m/TB_reports_risk_per_pixel.csv"

driver = ogr.GetDriverByName("ESRI Shapefile")
grid_ds = driver.Open(path_grid, 0)
grid_layer = grid_ds.GetLayer()

bite_ds = driver.Open(path_bites, 0)
bite_layer = bite_ds.GetLayer()

with open(path_out, "w", newline="") as w:
    writer = csv.writer(w, delimiter=";")
    writer.writerow(["cellid", "biteid"])
    for bite in bite_layer:
        bite_geom = bite.GetGeometryRef()
        j = 0
        rowid = int(bite.GetField(0))
        for cell in grid_layer:
            cell_geom = cell.GetGeometryRef()
            distance = int(bite_geom.Distance(cell_geom.Centroid()))
            if distance <= 1000:
                if bite_geom.Within(cell_geom):
                    newrow = [j, bite.GetField(0)]
                    writer.writerow(newrow)
                    break
                elif bite_geom.Intersects(cell_geom):
                    newrow = [j, bite.GetField(0)]
                    writer.writerow(newrow)
                    break
            w.flush()
            j += 1
        grid_layer.SetNextByIndex(0)

