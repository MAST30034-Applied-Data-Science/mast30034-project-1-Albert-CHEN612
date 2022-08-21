import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt
import pandas

nyc_boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))

taxi_6am=gpd.read_file('taxi_6am.csv')
taxi_zone=gpd.read_file('taxi_zones.csv')

taxi_zone['yellow_5pm']=taxi_6am['yellow_5pm']
taxi_zone['LocationID']=taxi_6am['LocationID']
taxi_zone['yellow_6am']=taxi_6am['yellow_6am']

tickets = gpd.read_file(gplt.datasets.get_path('nyc_parking_tickets'))
# print(tickets.head())

tickets=taxi_zone
tickets['yellow_5pm']=pandas.to_numeric(tickets['yellow_5pm'])
tickets['yellow_6am']=pandas.to_numeric(tickets['yellow_6am'])

proj = gcrs.AlbersEqualArea(central_latitude=40.7128, central_longitude=-74.0059)

def plot_state_to_ax(state, ax):
    gplt.choropleth(
        taxi_zone.set_index('LocationID').loc[:, [state, 'geometry']],
        hue=state, cmap='YlGn',
        legend=True,
        linewidth=0.0, ax=ax
    )
    gplt.polyplot(
        nyc_boroughs, edgecolor='black', linewidth=0.5, ax=ax
    )


f, axarr = plt.subplots(2, 2, figsize=(12, 7), subplot_kw={'projection': proj})

plt.suptitle('Yellow taxi pick-up times in March 1 2016 in NYC', fontsize=16)
plt.subplots_adjust(top=0.95)

plot_state_to_ax('yellow_5pm', axarr[0][0])
axarr[0].set_title('5pm')

plot_state_to_ax('yellow_6am', axarr[0][1])
axarr[1].set_title('6am')

plot_state_to_ax('pa', axarr[1][0])
axarr[1][0].set_title('Pennsylvania (n=215,065)')

plot_state_to_ax('ct', axarr[1][1])
axarr[1][1].set_title('Connecticut (n=126,661)')

plt.savefig("nyc-parking-tickets.png", bbox_inches='tight')
