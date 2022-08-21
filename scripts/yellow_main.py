import pandas as pd
from pandas import read_parquet
import os
import matplotlib.pyplot as plt


# loading all taxi data from dir and calculate the total count of trips.

def dataloading(dirpath):
    filenamelist = os.listdir(dirpath)
    ncount = 0
    data = dict()
    for filename in filenamelist:
        print(filename)
        datatmp = read_parquet(dirpath + filename)
        ncount = ncount + datatmp.count()
        print(datatmp.head())
        data[filename] = datatmp
    return data, ncount


# calculating the trip frequency per hour in one day
def trips_time_frequency(parquet_data):
    times = parquet_data['tpep_pickup_datetime']
    hours = dict()
    for i in times:
        hour = int(str(i)[11:13])
        if hour in hours:
            hours[hour] = hours[hour] + 1
        else:
            hours[hour] = 1
    print(hours)
    print(hours.values())


def trips_nyc_loc_frequency(data, time):
    dataloc = data['PULocationID']
    times = data['tpep_pickup_datetime']
    locs = dict()
    for i in range(len(dataloc)):
        loc = int(dataloc[i])
        hour = int(str(times[i])[11:13])
        print(hour)
        print(time)
        if hour == time:
            if loc in locs:
                locs[loc] = locs[loc] + 1
            else:
                locs[loc] = 1
    print(locs)
    print(locs.values())
    return locs


def trips_nyc_loc_frequency_times(data, time_seg):
    dataloc = data['PULocationID']
    times = data['tpep_pickup_datetime']
    locs = dict()
    for i in range(len(dataloc)):
        loc = int(dataloc[i])
        hour = int(str(times[i])[11:13])
        print(hour)
        if hour in time_seg:
            if loc in locs:
                locs[loc] = locs[loc] + 1
            else:
                locs[loc] = 1
    print(locs)
    print(locs.values())
    return locs


# def matrix_construction(data, threshold_num):
#     dataloc = data['PULocationID']
#     times = data['tpep_pickup_datetime']
#     locs = dict()
#     for i in range(len(dataloc)):
#         loc = int(dataloc[i])
#         hour = int(str(times[i])[11:13])
#         print(hour)
#         print(time)
#         if hour == time:


def time_segmentation(data):
    times = data['lpep_pickup_datetime']
    dataloc_pu = data['PULocationID']
    dataloc_dr = data['DOLocationID']
    mp=[]
    dt=[]
    ep=[]
    nt=[]
    for i in range(len(dataloc_pu)):
        loc_pu = int(dataloc_pu[i])
        loc_dr = int(dataloc_dr[i])
        hour = int(str(times[i])[11:13])
        if hour >=6 and hour <9:
            mp.append((loc_pu, loc_dr))
        elif hour >=9 and hour<17:
            dt.append((loc_pu, loc_dr))
        elif hour >= 17 and hour < 21:
            ep.append((loc_pu, loc_dr))
        elif hour > 17 or hour <6:
            nt.append((loc_pu, loc_dr))

    all=dict()
    all['mp']=mp
    all['dt'] = dt
    all['ep'] = ep
    all['nt'] = nt
    return all


def loc_frequency_seg(data_list):
    locs = dict()

    for i in range(len(data_list)):
        loc_pu = data_list[i][0]
        loc_dr = data_list[i][1]
        if loc_pu in locs:
            locs[loc_pu] = locs[loc_pu] + 1
        else:
            locs[loc_pu] = 1
    print(locs)
    print(locs.values())



# data, ncount = dataloading('green/test/')
data, ncount = dataloading('yellow/test/')

import networkx as nx

print(ncount)

for filename in data:

    # green_locs=trips_nyc_loc_frequency(data[filename], 6)
    # print(green_locs)
    # for i in range(266):
    #     if i in green_locs:
    #         pass
    #     else:
    #         green_locs[i]=0
    #
    # print(green_locs)
    # times=[6,7,8]
    # times = [9,10,11,12,13,14,15,16]
    # times = [17,18,19]
    times = [20,21,22,23,0,1,2,3,4,5]

    green_locs = trips_nyc_loc_frequency_times(data[filename], times)
    print(green_locs)
    for i in range(266):
        if i in green_locs:
            pass
        else:
            green_locs[i] = 0

    print(green_locs)



    #
    # # all_data=time_segmentation(data[filename])
    # print(all_data)
    #
    # res=dict()
    #
    # for i in all_data['mp']:
    #     print(i)
    #     if i in res:
    #         res[i]=res[i]+1
    #     else:
    #         res[i]=1
    #     # res[str(i)]=all_data['mp'].count(i)

    # print(res)
