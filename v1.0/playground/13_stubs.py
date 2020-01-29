# plt.figure(figsize=(10, 6))
# ax = plt.axes(projection=ccrs.epsg(28992))

axpoi = plt.subplot(1, 2, 1, projection=ccrs.epsg(28992))
axpoi.add_feature(cfeature.LAND.with_scale('50m'))
axpoi.add_feature(cfeature.OCEAN.with_scale('50m'))
axpoi.add_feature(cfeature.LAKES.with_scale('50m'))
axpoi.add_feature(cfeature.BORDERS.with_scale('50m'))
axpoi.coastlines('50m')
# axpoi.set_extent(NL_EXT)

ban = l[0].GetRasterBand(1).ReadAsArray()
ban[ban<0] = np.nan

axpoi.imshow(ban, origin="upper", transform=ccrs.epsg(28992), interpolation="None")

plt.show()

#
# # axpoi.imshow(np.roll(np.roll(ban, shift=-7, axis=1), shift=-13, axis=0), origin="upper", extent=NL_EXT, interpolation="None")
#

#
# print("hola")
#
# axalg = plt.subplot(1, 2, 2, projection=ccrs.epsg(28992))
# axalg.imshow(ban, origin="upper", extent=NL_EXT, interpolation="None")
#
# plt.show()
#
# #
# # ax.imshow(ban, origin="upper", extent=NL_EXT, interpolation="None")
# # plt.show()
#
# # ax.imshow(l[0].GetRasterBand(1).ReadAsArray(), origin="lower", transform=ccrs.EuroPP(), zorder=10)
# # plt.show()

