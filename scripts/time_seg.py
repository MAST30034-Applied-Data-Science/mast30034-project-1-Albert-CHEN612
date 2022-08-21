import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt
import pandas

nyc_boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))

taxi_6am=gpd.read_file('green_time_seg_per.csv')
taxi_zone=gpd.read_file('taxi_zones.csv')

# green_6to8	green_9to16	green_17to19	green_20to5	LocationID

taxi_zone['green_6to8']=taxi_6am['green_6to8']
taxi_zone['LocationID']=taxi_6am['LocationID']
taxi_zone['green_9to16']=taxi_6am['green_9to16']
taxi_zone['green_17to19']=taxi_6am['green_17to19']
taxi_zone['green_20to5']=taxi_6am['green_20to5']


tickets=taxi_zone
tickets['green_6to8']=pandas.to_numeric(tickets['green_6to8'])
tickets['green_9to16']=pandas.to_numeric(tickets['green_9to16'])
tickets['green_17to19']=pandas.to_numeric(tickets['green_17to19'])
tickets['green_20to5']=pandas.to_numeric(tickets['green_20to5'])



proj = gcrs.AlbersEqualArea(central_latitude=40.7128, central_longitude=-74.0059)

def plot_state_to_ax(state, ax, cmaps):
    gplt.choropleth(
        taxi_zone.set_index('LocationID').loc[:, [state, 'geometry']],
        hue=state, cmap=cmaps,
        legend=True,
        linewidth=0.0, ax=ax
    )
    gplt.polyplot(
        nyc_boroughs, edgecolor='black', linewidth=0.5, ax=ax
    )


f, axarr = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': proj})

# plt.suptitle('Yellow taxi pick-up times in March 1 2016 in NYC', fontsize=16)
plt.subplots_adjust(top=0.95)

plot_state_to_ax('green_6to8', axarr[0][0],'Greens')
axarr[0][0].set_title('6am to 8am')

plot_state_to_ax('green_9to16', axarr[0][1],'Greens')
axarr[0][1].set_title('9am to 4pm')

plot_state_to_ax('green_17to19', axarr[1][0], 'Greens')
axarr[1][0].set_title('5pm to 7pm')

plot_state_to_ax('green_20to5', axarr[1][1],'Greens')
axarr[1][1].set_title('8pm to 5am')

plt.savefig("green_time_seg.png", bbox_inches='tight')

# plot_state_to_ax('pa', axarr[1][0])
# axarr[1][0].set_title('Pennsylvania (n=215,065)')

# plot_state_to_ax('ct', axarr[1][1])
# axarr[1][1].set_title('Connecticut (n=126,661)')

plt.savefig("nyc-parking-tickets.png", bbox_inches='tight')
