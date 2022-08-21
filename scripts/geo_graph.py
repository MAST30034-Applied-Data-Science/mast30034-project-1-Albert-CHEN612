import geopandas
import geoplot
import pandas as pd
# from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

store_locDF = pd.read_csv('stores.csv')
cols_to_keep = ['store_longitude', 'store_latitude']

crs = {'init': 'epsg:4326'}

boroughs = geopandas.read_file(geoplot.datasets.get_path('nyc_boroughs'))

geo_df = geopandas.GeoDataFrame(store_locDF, crs=crs, geometry=geopandas.points_from_xy(store_locDF["store_longitude"],
                                                                                        store_locDF["store_latitude"]))

fig, ax = plt.subplots(figsize=(15, 15))
boroughs.plot(ax=ax, alpha=0.4, color="grey")
geo_df[geo_df['active_yn'] == "Y"].plot(ax=ax, markersize=200, alpha=0.4, color="green", label="Active Store")
geo_df[geo_df['active_yn'] == "N"].plot(ax=ax, markersize=200, alpha=0.4, color="red", label="Inactive Stores")
plt.legend()

