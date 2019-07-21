from math import radians, sin, cos, sqrt, asin
from sklearn.cluster import KMeans
import pandas as pd
import geocoder
import numpy as np
import csv
from collections import Counter
from operator import itemgetter
import MySQLdb
from scipy.spatial import distance

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 



from django.conf import settings
from django.db.models import Max
from cutter.models import voter_json, region, intersections, region_progress


from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

import os
import time
import json
import re
import pymysql
import sqlalchemy
import copy


import httplib2

from googleapiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
#from google.oauth2 import service_account
import base64
#from googleapiclient import errors, discovery
from pyzillow.pyzillow import ZillowWrapper, GetDeepSearchResults


from .email import *
from .output import *


#Take a row with "LAT" and "LON" and then kwargs 'lat1' and 'lon1'
#And calculate teh haversine distance between the 2 points
def add_distance(row,**kwargs):
    #print kwargs
    lat2 = row["LAT"]
    lon2 = row["LON"]
    distance = haversine(kwargs['lat1'],kwargs['lon1'],lat2,lon2)
    return distance


#Get the distance in km between 2 lat and lon points
def haversine(lat1, lon1, lat2, lon2):
 
  R = 6372.8 # Earth radius in kilometers
 
  dLat = radians(lat2 - lat1)
  dLon = radians(lon2 - lon1)
  lat1 = radians(lat1)
  lat2 = radians(lat2)
 
  a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
  c = 2*asin(sqrt(a))
 
  return R * c


#Take the data (master list of addresses)
#And the canvas_location (central address)
#And find the num_turfs * turf_sizes closest addresses
#Save it as new_file_name
def make_filtered_file(data,canvas_location,num_turfs,turf_sizes,new_file_name):
    #Get the coordinates of the canvas central point
    coords = get_coordinates(canvas_location,False)
    lat1 = coords[0]
    lon1 = coords[1]
    #Add a column to the list of addresses
    #That measures the distance from the center location
    data["distances"] = data.apply(add_distance,axis=1,lat1=lat1,lon1=lon1)
    #Sort the addresses by this distance
    data = data.sort_values("distances")
    #Cut the data to get the closest n addresses
    slice_data = data[:int(1.5*num_turfs * turf_sizes)]
    #Save them as an Excel file
    slice_data.to_excel(new_file_name)



def get_coordinates(address,check_intersect=True):
    g = geocoder.arcgis(address)
    if check_intersect:
        if not g.address:
            return []
        elif "&" in g.address:
            return list(g.latlng)
        else:
            return []
    return list(g.latlng)


#Open the list of intersections
#Load them into a dictionary
#Do it for both - so if 12th Street and Walnut Street intersect
#threshold_dict['12th Street']['Walnut Street'] will be an entry
#As will threshold_dict['Walnut Street']['12th Street']
def load_threshold_dict(region,lon_only=False):
    #intersect_data = pd.read_csv("Intersections_1.csv")
    intersect_data = read_mysql_data("""SELECT * from canvas_cutting.cutter_intersections where lat <> 0 and
         region = '{region}'""".format(region = region))
    if lon_only:
        intersect_data = intersect_data[intersect_data["lon"].notnull()]
    threshold_dict = {}
    for row in intersect_data.itertuples():
        s1 = row.street1
        s2 = row.street2
        if not s1 in threshold_dict:
            threshold_dict[s1] = {}
        threshold_dict[s1][s2] = None
        if not s2 in threshold_dict:
            threshold_dict[s2] = {}
        threshold_dict[s2][s1] = None
    return threshold_dict


#Take a list of data and get a list of all potential intersections
def get_potential_intersections(slice_data,region,km_threshold = .1,lat_threshold=.0008):
    #Load the list of already checked intersections
    threshold_dict = load_threshold_dict(region)
    threshold_list = {}

    #Scroll through all through potential combinations of addresses
    #Basically we're trying to estimate which streets might intersect
    #We do this by finding pairs of addresses that are close to each other
    #But on different streets
    #This is very computationally expensive
    #So let's see if we can improve it
    for i,row in enumerate(slice_data.itertuples()):
        for j,test_row in enumerate(slice_data.itertuples()):
            #Skip if the addresss are the same
            if j<=i:
                continue
            #If we've already checked the intersection don't check again
            if row.STREET in threshold_dict and test_row.STREET in threshold_dict[row.STREET]:
                continue
            #If they're the same street don't bother checking
            if row.STREET == test_row.STREET:
                continue
            #If they're not close together don't bother checking
            if abs(row.LAT - test_row.LAT) > lat_threshold:
                continue
            #Check distance between the 2 points
            #If they're not close keep going
            distance = haversine(row.LAT,row.LON,test_row.LAT,test_row.LON)
            if distance > km_threshold:
                continue
            #Sort the streets for the purpose of writing them to the dictionary
            streets = tuple(sorted([row.STREET,test_row.STREET]))
            #If this is the first time we've checked the 2 streets set their distance as the distance between the streets
            if streets not in threshold_list:
                threshold_list[streets] = distance
            #Otherwise check if these points are the 2 closest points on these 2 particular streets
            elif distance < threshold_list[streets]:
                threshold_list[streets] = distance
    return threshold_list


#Write an intersection to the master list of intersections
def write_row(row,new_file=False):
    if new_file:
        f = open('Intersections_1.csv','wb')
    else:
        f = open('Intersections_1.csv','ab')
    writer = csv.writer(f)
    writer.writerow(row)
    f.close()

def write_row(row,region):
    i = intersections()
    i.region = region
    i.street1 = row[0]
    i.street2 = row[1]
    i.distance = row[2]
    if len(row) > 3:
        i.lat = row[3]
        i.lon = row[4]
    else:
        i.lat = 0
        i.lon = 0
    i.save()


#Check the thresholds for a list of addresses
def update_thresholds(slice_data,region):
    #Get the list of all potential intersections
    potential_intersections = get_potential_intersections(slice_data,region)
    num_geocodes=0
    err_count = 0
    #Scroll through the list of potential intersections
    for (i,j),k in potential_intersections.iteritems():
        num_geocodes+=1
        address = i + " and " + j +", " + region
        #Check Google's geocoder to see if it's a valid intersection
        #e.g. if we're checking 12th Street and Walnut Street
        #Check the coordinates of 12th Street & Walnut Street, Austin, TX
        #If it's not a valid intersection it won't return coordinates
        coords = get_coordinates(address)
        #Return an error message if 8 straight addresses show no intersection
        #This is a good sign that the API is maxed out
        if coords:
            err_count = 0
        else:
            err_count += 1
            if err_count >9999997:
                return False
        #Write the result of the check to the master list of intersections
        write_row([i,j,k] + coords,region)
        #Google gives us 1500 checks per day
        #If we run through them move on
        #And send a message saying to run again tomorrow
        #if num_geocodes > 1500:
            #return False
    return True
    

#Update the data to set the nulls to 0s
def update_slice_data_nulls(slice_data):
    slice_data.loc[slice_data["voters"].isnull(),"voters"] = 0
    slice_data.loc[slice_data["doors"].isnull(),"doors"] = 0
    return slice_data


#Add 2 columns to the list of addresses
#For the average lat and average lon of the whole street
#This will be used for the clustering algorithm
def update_slice_data_avgs(slice_data):
    avg_locs = pd.pivot_table(slice_data,values=("LAT","LON"),index=("STREET"),aggfunc=np.mean)
    avg_locs = avg_locs.rename_axis(None, axis=1).reset_index()
    avg_locs.columns = ["STREET","street_lat","street_lon"]
    avg_locs["street_lat"] = avg_locs["street_lat"]  
    avg_locs["street_lon"] = avg_locs["street_lon"] 
    slice_data_updated = slice_data.merge(avg_locs,on="STREET")
    return slice_data_updated


#Assign a cluster to each address
def update_slice_data_clusters(slice_data_updated,num_clusters):
    fit_data = slice_data_updated.loc[:,("LAT","LON","street_lat","street_lon")]
    #Use the kmeans algorithm to cluster each address into the specified # of clusters
    kmeans = KMeans(n_clusters=num_clusters).fit(fit_data)
    slice_data_updated['Cluster'] = kmeans.labels_
    return slice_data_updated


#Check that a clusters is a continuous route
#For each street check if it connects to every other street 
#Return a list of streets that don't connect
def check_cluster(df,cluster,threshold_dict):
    ret = []
    df = df[df["Cluster"]==cluster]
    df = df.reset_index()
    streets = df["STREET"].unique()
    for i in streets:
        count = checkstreet(i,streets,threshold_dict)
        if count > 0:
            ret.append([i,count])
    return ret


#For a street and a cluster
#Check if it connects to the other streets in the cluster
#using a modification of Dijkstra's algorithm
#Return the number of streets in the cluster it does not connect to
def checkstreet(street,streets,threshold_dict):
    ret = []
    unchecked = {i:999 for i in streets}
    checked = {}
    unchecked[street] = 0

    while unchecked:
        minval = 1000
        minkey = None
        for key,val in unchecked.iteritems():
            if val < minval:
                minval = val
                minkey = key

        checked[minkey] = unchecked[minkey]
        del unchecked[minkey]
        if not minkey in threshold_dict:
            continue
        to_check = set(unchecked.keys()).intersection(threshold_dict[minkey].keys())
        for i in to_check:
            distance = minval + 1
            unchecked[i] = min(distance,unchecked[i])

    for key,val in checked.iteritems():
        if val == 999:
            ret.append(street)

    return len(ret)


#Take a street that does not fit in its current cluster
#And try to find a new cluster for it
#Inputs are:
#street - name of the non-fitting street
#cur_cluster - number of the cluster it's currently in
#slice_data_updated - list of all points
#threshold_dict - list of all intersections
def new_cluster(street,cur_cluster,slice_data_updated,threshold_dict):
    #Take all addresses on the specified street in the specified cluster
    street_points = slice_data_updated.loc[(slice_data_updated["STREET"]==street) & (slice_data_updated["Cluster"]==cur_cluster),
                                ("LAT","LON")]
    #Find the mean lat and lon of that street
    street_avg = np.mean(street_points["LAT"]),np.mean(street_points["LON"])
    #Take the mean lat and lon of all the other clusters
    cluster_avgs = pd.pivot_table(slice_data_updated,index="Cluster",values=("LAT","LON"),aggfunc=np.mean)
    cluster_avgs = cluster_avgs.rename_axis(None, axis=1).reset_index()
    #Sort the rest of the clusters to find the ones closest to the specified street
    cluster_avgs["distance"] = cluster_avgs.apply(lambda row: haversine(row["LAT"],row["LON"],street_avg[0],street_avg[1]),axis=1)
    cluster_avgs = cluster_avgs.sort_values("distance")
    #Scroll through the 4 closest clusters and see if the street fits in any of them
    for count,row in enumerate(cluster_avgs.itertuples()):
        if count > 4:
            break
        #Obviously don't put the street in its same cluster
        if row.Cluster == cur_cluster:
            continue
        #df will be all the addresses in the cluster we're checking
        df = slice_data_updated[slice_data_updated["Cluster"]==row.Cluster]
        df = df.reset_index()
        #Get a list of all the streets in the cluster we're checking
        #And make sure they all connect to the street we're moving the new street to
        #If they do move the street to that cluster
        streets = df["STREET"].unique()
        if checkstreet(street,streets,threshold_dict) == 0:
            return row.Cluster
    return None


#Scroll through each cluster and remove the streets that don't connect
#Then try to move those streets to a new cluster
def update_slice_data_check_clusters(slice_data_updated,num_clusters,threshold_dict):
    #Scroll through each cluster
    for i in range(num_clusters):
        #result will be a list of streets in that cluster that don't connect to others
        result = check_cluster(slice_data_updated,i,threshold_dict)
        #If they all connect result will be blank
        #And we can continue to the next cluster
        if not result:
            continue
        #If there are streets that don't connect work through them
        while result:
            #Take the street that doesn't connect to the most other streets
            bad_street = sorted(result,key=lambda k: k[1],reverse=True)[0][0]
            #See if we can find a new cluster for that ill-fitting street
            upd_cluster = new_cluster(bad_street,i,slice_data_updated,threshold_dict)
            #If we find a new home for that street move it to that street
            #If not put it in cluster -1, which is where we keep all the streets that couldn't find a home
            if upd_cluster:
                slice_data_updated.loc[(slice_data_updated.STREET==bad_street) & (slice_data_updated.Cluster == i), "Cluster"] = upd_cluster
            else:
                slice_data_updated.loc[(slice_data_updated.STREET==bad_street) & (slice_data_updated.Cluster == i), "Cluster"] = -1
            #Check again for non-fitting streets
            #If none fit it will break the while loop and we move to the next cluster
            #If there are still streets that don't fit run this process again
            result = check_cluster(slice_data_updated,i,threshold_dict)
    return slice_data_updated
        


#Take all streets that were removed from clusters that couldn't find a new one
#And see if we can put them in a new cluster
#Checking if it connects to all the other streets
def check_bad_streets(slice_data_updated,threshold_dict):
    #Cluster -1 is all the streets that aren't a cluster
    bad_streets = slice_data_updated.loc[slice_data_updated.Cluster==-1,"STREET"].unique()
    for bad_street in bad_streets:
        #Run the new_cluster function which will return a new cluster or None
        upd_cluster = new_cluster(bad_street,-1,slice_data_updated,threshold_dict)
        #If it returns, update the street to put those addresses in a new cluster
        if upd_cluster:
            slice_data_updated.loc[(slice_data_updated.STREET==bad_street) & (slice_data_updated.Cluster == -1),"Cluster"] = upd_cluster
        else:
            num_doors = sum(slice_data_updated.loc[(slice_data_updated.STREET==bad_street) & (slice_data_updated.Cluster == -1),"doors"])
            if num_doors > 40:
                max_cluster = max(slice_data_updated["Cluster"])
                slice_data_updated.loc[(slice_data_updated.STREET==bad_street) & (slice_data_updated.Cluster == -1),"Cluster"] = max_cluster + 1
    return slice_data_updated



def check_split(filt_data):
    filt_data = filt_data.sort_values('doors',ascending=False).reset_index()
    max_doors = filt_data.loc[0,"doors"]
    sum_doors = sum(filt_data["doors"])
    if sum_doors - max_doors < 80:
        return False
    else:
        return True

#Check the distance between the 2 furthest apart addresses on the same street
def row_walking_distance(row):
    return haversine(row["latmin"],row["lonmin"],row["latmax"],row["lonmax"])

#Function to approximate the walking distance for a turf/cluster
#The formula is for each street takes the max and min lat and lon and gets the distance
#This will underestimate the walking distance for curvy roads
#But provides a decent estimate
def get_walking_distance(row,**kwargs):
    #Pass the list of data as a kwarg in the apply function
    slice_data_updated = kwargs['slice_data_updated']
    #df gets all the addresses in a specified cluster
    df = slice_data_updated[slice_data_updated['Cluster']==int(row['Cluster'])]
    #Get the min and max lat and lon for each street in the cluster
    f = {"LAT": ["min","max"],"LON": ["min","max"]}
    t = df.groupby(['STREET'],as_index=False).agg(f)
    t.columns = ["STREET","latmin","latmax","lonmin","lonmax"]
    #Sum the walking distance for each street in the turf
    return sum(t.apply(row_walking_distance,axis=1))


#Get row distance from a specified point
def row_distance(row,**kwargs):
        center_lat = kwargs['center_lat']
        center_lon = kwargs['center_lon']
        return haversine(row["LAT"],row["LON"],center_lat,center_lon)

#Update clusters to be ordered by distance from the center point
def update_cluster_numbers(slice_data_updated):
    #Remove the non-clusered cluster
    eligible_data = slice_data_updated[slice_data_updated["Cluster"]!=-1]
    #Take the average lat and lon of each cluster
    cluster_avgs = eligible_data.groupby("Cluster",as_index=False)["LAT","LON"].mean()
    #Make the center lat and center lon the address closest to the target
    [center_lat,center_lon] = eligible_data.iloc[0,:]["LAT"],eligible_data.iloc[0,:]["LON"]
    #Make a new column for each cluster - distance = it's center's distance from the overall center
    cluster_avgs["distance"] = cluster_avgs.apply(row_distance,axis=1,center_lat=center_lat,center_lon=center_lon)
    #Sort by distance from the center
    cluster_avgs = cluster_avgs.sort_values("distance")
    cluster_avgs = cluster_avgs.reset_index(drop=True)
    cluster_avgs = cluster_avgs.reset_index()
    merge_data = cluster_avgs.loc[:,("index","Cluster")]
    #Merge the new clusters with the data
    slice_data_updated = slice_data_updated.merge(merge_data,on="Cluster",how="left")
    #Delete the old cluster columns and name the added column Cluster
    slice_data_updated = slice_data_updated.drop('Cluster',1)
    slice_data_updated.columns = list(slice_data_updated.columns[:-1]) + ['Cluster']
    #Set the addresses not in a cluster to cluster -1
    slice_data_updated.loc[pd.isnull(slice_data_updated["Cluster"]),"Cluster"] = -1
    slice_data_updated["Cluster"] = slice_data_updated["Cluster"].astype(int)
    return slice_data_updated


#Take the list of addresses and return a list of turfs with stats about each turf
def get_cluster_totals(slice_data_updated):
    #Remove the addresses not assigned to a turf
    eligible_data = slice_data_updated[slice_data_updated["Cluster"]!=-1]
    #Take the average lat and lon, the number of addresses, voters and doors
    f = {"voters": ['sum',len],"doors": 'sum', "LAT": 'mean',"LON": "mean"}
    cluster_totals = eligible_data.groupby("Cluster",as_index=False).agg(f)
    cluster_totals["Cluster"] = cluster_totals["Cluster"].astype(int)
    [center_lat,center_lon] = slice_data_updated.iloc[0,:]["LAT"],slice_data_updated.iloc[0,:]["LON"]
    #Calculate the distance from the center
    cluster_totals["distance"] = cluster_totals.apply(row_distance,axis=1,center_lat=center_lat,center_lon=center_lon)
    #Calculate teh walking distance for each cluster
    cluster_totals["walking_distance"] = cluster_totals.apply(get_walking_distance,slice_data_updated=slice_data_updated,axis=1)
    cluster_totals.columns = ["Cluster","latmean","lonmean","doors","voters","addresses","distance","walking_distance"]
    return cluster_totals


#Split a cluster into 2
#Inputs are:
#df = total datat
#cluster = the # of the cluster we're splitting
#splits - is the number of splits (right now we're only doing 2)
def split_cluster(df,cluster,splits,min_size=60):
    #Make a copy so we don't endit it
    df = df.copy()
    #Filter for the addresses in the specified cluster
    filt_data = df.loc[df["Cluster"]==cluster,("LAT","LON","street_lat","street_lon","Cluster")].copy()
    
    #if not check_split(filt_data):
    #    return df
    #Get the highest numbered cluster
    max_cluster = max(df["Cluster"])
    split_counts = []
    for i in range(5):
        #Use kmeans to split the cluster
        kmeans = KMeans(n_clusters=splits,init='random').fit(filt_data)
        counter = Counter(kmeans.labels_)
        small_cluster_size = min(counter[0],counter[1])
        split_counts.append([small_cluster_size,kmeans.labels_])
    labels = max(split_counts,key=itemgetter(0))[1]
    #Add the column label depending on where k-means put it
    filt_data["label"] = labels
    #print filt_data
    #Make the new clusters created by kmeans into new clusters
    for i in range(1,splits):
        filt_data.loc[filt_data["label"]==i,"Cluster"] = max_cluster + i
    df.update(filt_data["Cluster"])
    return df



def top_street(slice_data_updated,cluster):
    filt_data = slice_data_updated.loc[slice_data_updated["Cluster"]==cluster,:]
    agg_data = filt_data.groupby("STREET").agg(len).reset_index()
    street = agg_data.loc[0,"STREET"]
    return street


#Take a cluster and see if i tcan be merged with another cluster
#inputs are:
#cluster_totals = list of all clusters
#slice_data_updated = list of all addresses
#cluster = #of the cluster we're trying to merge
#threshold_dict = list of all intersections
#turf_size - desired turf size
#missing clusters - clusters that already got merged - don't try to merge with them
def new_whole_cluster(cluster_totals,slice_data_updated,cluster,threshold_dict,turf_size,missing_clusters):
    #Get the # of doors in the specified cluster
    cluster_doors = cluster_totals.loc[cluster_totals["Cluster"]==cluster,"doors"]
    cluster_doors = cluster_doors.iloc[0]
    #Get the walking distance of the specified cluster
    cluster_walking_distance = cluster_totals.loc[cluster_totals["Cluster"]==cluster,"walking_distance"]
    cluster_walking_distance = cluster_walking_distance.iloc[0]
    #Get the mean lat and lon of the specified cluster
    cluster_lat = cluster_totals.loc[cluster_totals["Cluster"]==cluster,"latmean"].iloc[0]
    cluster_lon = cluster_totals.loc[cluster_totals["Cluster"]==cluster,"lonmean"].iloc[0]
    street_avg = cluster_lat,cluster_lon
    #Get the average lat and lon of each cluster
    #And sort by distance to the cluster we're trying to merge
    cluster_avgs = pd.pivot_table(slice_data_updated,index="Cluster",values=("LAT","LON"),aggfunc=np.mean)
    cluster_avgs = cluster_avgs.rename_axis(None, axis=1).reset_index()
    cluster_avgs["distance"] = cluster_avgs.apply(lambda row: haversine(row["LAT"],row["LON"],street_avg[0],street_avg[1]),axis=1)
    cluster_avgs = cluster_avgs.sort_values("distance")
    #Get the most common street in the cluster we're trying to merge
    street = top_street(slice_data_updated,cluster)
    #Scroll through all other clusters
    for count,row in enumerate(cluster_avgs.itertuples()):
        #Only try the 4 clusters closest to the specified cluster
        if count > 4:
            break
        #Skip the missing clusters list
        if row.Cluster in missing_clusters:
            continue
        #Cluster -1 isn't an actual cluster
        if row.Cluster == -1:
            continue
        #Don't merge a cluster with itself
        if row.Cluster == cluster:
            continue
        #Filter the data for the cluster we're scrolling through    
        df = slice_data_updated[slice_data_updated["Cluster"]==row.Cluster]
        df_doors = sum(df["doors"])
        #Get walking distance of the specified cluster
        walking_distance = cluster_totals.loc[cluster_totals["Cluster"]==row.Cluster,"walking_distance"]
        walking_distance = walking_distance.iloc[0]

        #Use the formula to find the max numbers of doors in a cluster
        max_doors = 2.0 * turf_size - 25 * walking_distance - 25 * cluster_walking_distance
        #If merging these 2 clusters if over the max # of doors we can't merge
        if df_doors + cluster_doors >= max_doors:
            continue
        df = df.reset_index()
        #Get unique streets in 2nd cluster
        streets = df["STREET"].unique()
        #If main street connects to all streets in the proposed cluster they can merge
        if checkstreet(street,streets,threshold_dict) == 0:
            return row.Cluster
    return None


#Get a list of streets and ranges of address in a specified turf
def get_street_list(df):
    streets = pd.pivot_table(df,index="STREET",values="NUMBER",aggfunc=(min,max))
    streets = streets.rename_axis(None, axis=1).reset_index()
    ret = []
    for row in streets.itertuples():
        ret.append(str(row.min) + " " + row.STREET + " to " + str(row.max) + " " + row.STREET)
    return ret


#Figure out how many rows to make 
#To be safe we double the number of registered doors at an address
#i.e. if an apartment has 14 registered doors we leave 28 spaces
#as not every resident will be registered
def num_rows(row):
    return min(2,2*int(row.doors)/2)


def make_json_columns(df,stand_alone_columns,json_columns):
    df = df.copy()
    df['json_col'] = df[json_columns].apply(lambda x: x.to_dict(), axis=1)
    print 'added json'
    df = df.loc[:,stand_alone_columns + ('json_col')]
    return df

def make_mysql_connection(skip_db = False):
    user = settings.DATABASES['default']['USER']
    password = settings.DATABASES['default']['PASSWORD']
    database_name = settings.DATABASES['default']['NAME']
    if skip_db:
        return MySQLdb.connect(user=user,password=password)
    return MySQLdb.connect(user=user,password=password,database_name=database_name)

def make_sqlalchemy_connection():
    user = settings.DATABASES['default']['USER']
    password = settings.DATABASES['default']['PASSWORD']
    return sqlalchemy.create_engine('mysql+pymysql://{user}:{password}@localhost:3306/canvas_cutting'.format(user=user,password=password))

def make_mysqldb_connection():
    x = MySQLdb.connect(host='localhost', \
                    user=settings.DATABASES['default']['USER'], \
                    passwd=settings.DATABASES['default']['PASSWORD'], \
                    db="canvas_cutting")
    x = MySQLdb.connect(host="159.89.185.59", \
                    user="username", \
                    passwd="password", \
                    db="canvas_cutting")
    return x

def execute_mysql(statement):
    conn = make_mysql_connection(True)
    c=conn.cursor()
    c.execute(statement)

def simple_query(query):
    conn = make_mysql_connection(True)
    c=conn.cursor()
    c.execute(query)
    return c.fetchall()

def write_mysql_data(df,table_name,region,if_exists='append',better_append=False,chunksize=None,dtype=None):
    df["region"] = region
    con = make_sqlalchemy_connection()
    if not better_append:
        df.to_sql(con=con, name=table_name, if_exists=if_exists,index=False,chunksize=chunksize,dtype=dtype)
    else:
        max_id = read_mysql_data("SELECT MAX(id) FROM {table_name}".format(table_name=table_name))
        id_range = range(max_id+1,max_id + len(df) + 1)
        df.to_sql(con=con, name=table_name, if_exists=if_exists,index=False,index_label=id_range) 

def read_mysql_data(query,alchemy=True):
    if alchemy:
        con = make_sqlalchemy_connection()
    else:
        con = make_mysqldb_connection()
    return pd.read_sql(query,con)

def write_json_data(df,json_columns,region):
    print 'starting function'
    for i in df.itertuples():
        try:
            vj = voter_json()
            vj.region = region
            vj.address = i.address
            temp_dict = {j: getattr(i,j) for j in json_columns}
            vj.json_data = json.dumps(temp_dict)
            vj.save()
        except Exception as e:
            print e
            print i
            1/0
            #continue

def cutoff(address,fin):
    f = address.find(fin)
    if f > -1:
        return address[:f]
    else:
        return address

def clean_dataframe(df):
    df["address"] = df.apply(clean_address,axis=1)
    return df

def clean_address(row):
    address = str(row.address)
    ret = address.upper()
    ret = ret.replace("DR.","DR")
    ret = ret.replace("DRIVE","DR")
    cutoffs = [",",", APT"," APT"," #", ", #"]
    for i in cutoffs:
        ret = cutoff(ret,i)
    return ret

def update_address(address,data,points,region):
    try:
        coords = get_coordinates(str(address) + ", " + region,False)
        ind = np.argpartition(distance.cdist([coords],points),4)
        data = data.reset_index()
        ret = [data.loc[ind[0][i],"address"] for i in range(4)]
        return ret
    except:
        return address


def iterate_merge(geocode_data,new_addresses,address_function,filename = None,**kwargs):
    if address_function:
        new_addresses = address_function(new_addresses,**kwargs)
    else:
        new_addresses = new_addresses
    merge_data = new_addresses.merge(geocode_data,on="address",how="left")
    good_data = merge_data[pd.notnull(merge_data["LAT"])]
    bad_data = merge_data[pd.isnull(merge_data["LAT"])]
    merge_perc = 1.0*len(good_data)/len(merge_data)
    good_data_merging = good_data.loc[:,new_addresses.columns]
    geo_merge_data = geocode_data.merge(good_data_merging,on="address",how="left")
    bad_geo_data = geo_merge_data[pd.isnull(geo_merge_data["full_street"])]
    bad_street_count = bad_data.groupby("full_street")["address"].agg(len)
    bad_street_count = bad_street_count.reset_index()
    worst_streets = bad_street_count.sort_values("address",ascending=False).iloc[:100,:]
    bad_geo_street_count = bad_geo_data.groupby("STREET")["address"].agg(len)
    bad_geo_street_count = bad_geo_street_count.reset_index()
    worst_geo_streets = bad_geo_street_count.sort_values("address",ascending=False).iloc[:100,:]
    #output = pd.concat([worst_streets.reset_index(drop=True), bad_geo_street_count.reset_index(drop=True)], axis=1)
    #output.columns = ["new_street","num missing","geocode_street","num_missing"]
    output = pd.concat([worst_streets.reset_index(drop=True), bad_geo_street_count.reset_index(drop=True)], axis=1,ignore_index=True)
    output.columns = ["new_street","num missing","geocode_street","num_missing"]
    ret = {}
    ret["good_data"] = good_data
    ret["bad_data"] =bad_data.loc[:,new_addresses.columns]
    ret["bad_geo_data"] = bad_geo_data.loc[:,geocode_data.columns]
    ret["bad_full_geo_data"] = bad_geo_data
    ret["merge_perc"] = merge_perc
    if filename:
        output.to_excel(filename)
    return ret

def address_head(row):
    address = str(row.address)
    ind = address.find(" ")
    ret = address[:ind]
    return re.sub("[^0-9]", "", ret)


def get_street_change_recs(df,**kwargs):
    ret = []
    geocode_data = kwargs["geocode_data"]
    region = kwargs["region"]
    df["address_head"] = df.apply(address_head,axis=1)
    df["update"] = 0
    points = np.array(geocode_data.loc[:,("LAT","LON")])
    unique_tails = sorted(df["full_street"].unique())
    
    for i in unique_tails:
        filt_data = df.loc[df["full_street"]==i,:]
        break_point = False
        for j in filt_data.itertuples():
            test_addresses = update_address(j.address,geocode_data,points,region)
            for k in test_addresses:
                ind = k.find(" ")
                if k[:ind] == j.address_head:
                    print j.full_street,k[ind+1:]
                    ret.append([j.full_street,k[ind+1:]])
                    break_point = True
                    break
            if break_point:
                break
    return ret

def write_csv(filename,row):
    f = open(filename,'ab')
    csvw = csv.writer(f)
    csvw.writerow(row)
    f.close()


def check_tail(tail,df,geocode_data,points,region):
    filt_data = df.loc[df["full_street"]==tail,:]
    break_point = False
    for j in filt_data.itertuples():
        test_addresses = update_address(j.address,geocode_data,points,region)
        for k in test_addresses:
            ind = k.find(" ")
            if k[:ind] == j.address_head:
                write_csv("Change_recs_{region}.csv".format(region=region),[j.full_street,k[ind+1:]])
                #ret.append([j.full_street,k[ind+1:]])
                break_point = True
                break
        if break_point:
            break


def check_tail_wrapper(args):
    return check_tail(*args)

def get_street_change_recs(df,**kwargs):
    ret = []
    geocode_data = kwargs["geocode_data"]
    region = kwargs["region"]
    df["address_head"] = df.apply(address_head,axis=1)
    df["update"] = 0
    points = np.array(geocode_data.loc[:,("LAT","LON")])
    unique_tails = sorted(df["full_street"].unique())
    tail_tuples = [(i,df,geocode_data,points,region) for i in unique_tails]
    pool = ThreadPool(2)
    pool.map(check_tail_wrapper,tail_tuples)

def get_coverage_ratio(region):
    good_data = simple_query("""SELECT COUNT(*) FROM canvas_cutting.cutter_canvas_data where region = '{region}' and 
         full_street <> ''""".format(region=region))[0][0]
    print good_data
    bad_data = simple_query("""SELECT COUNT(*) FROM canvas_cutting.cutter_bad_data where region = '{region}'""".format(region=region))[0][0]
    print bad_data
    return 1.0 * good_data / (good_data + bad_data)


def add_new_region(region_name):
    query_check = region.objects.filter(name = region_name)
    if query_check:
        return False
    region.objects.create(name=region_name)
    region_progress.objects.create(name=region_name)
    return True


def replace_list(df,**kwargs):
    filename = kwargs['replace_file']
    combos = pd.read_csv(filename)
    combos.columns = ['old_address','new_address']
    for row in combos.itertuples():
        df.loc[df["full_street"]==row.old_address,"address"] = \
            df.loc[df["full_street"]==row.old_address,"address"].apply(lambda s: s.replace(row.old_address,row.new_address))
    return df


def matchmaker(d1,d2):
    guys = sorted(d1.keys())
    gals = sorted(d2.keys())
    guysfree = guys[:]
    engaged  = {}
    guyprefers2 = copy.deepcopy(d1)
    galprefers2 = copy.deepcopy(d2)
    while guysfree:
        guy = guysfree.pop(0)
        guyslist = guyprefers2[guy]
        gal = guyslist.pop(0)
        fiance = engaged.get(gal)
        if not fiance and fiance != 0:
            # She's free
            engaged[gal] = guy
            #print("  %s and %s" % (guy, gal))
        else:
            # The bounder proposes to an engaged lass!
            galslist = galprefers2[gal]
            if galslist.index(fiance) > galslist.index(guy):
                # She prefers new guy
                engaged[gal] = guy
                #print("  %s dumped %s for %s" % (gal, fiance, guy))
                if guyprefers2[fiance]:
                    # Ex has more girls to try
                    guysfree.append(fiance)
            else:
                # She is faithful to old fiance
                if guyslist:
                    # Look again
                    guysfree.append(guy)
    return engaged


def mile_distance(row,**kwargs):
    lat = kwargs["coords"][0]
    lon = kwargs["coords"][1]
    dist = ((row["LAT"] - lat)**2 + (row["LON"] - lon)**2)**.5
    return 69* dist


def add_score(row,**kwargs):
    row["score"] = -1 * row["distance"]
    if row["Unit_Count"] > 0:
        row["score"] += .5* min(row["registered_voters"] / row["Unit_Count"],1)
        row["score"] += .5* min(row["16_general_voters"] / row["Unit_Count"],1)
        row["score"] += 1.25* min(row["16_primary_first_timers"] / row["Unit_Count"],1)
    return row["score"]
    return 69* dist


def remove_units(afford_units,skip_addresses):
    col = skip_addresses.columns[0].upper()
    skip_addresses = skip_addresses.rename(columns = {skip_addresses.columns[0]: col})
    afford_columns = afford_units.columns
    skip_addresses[col] = skip_addresses[col].str.upper()
    afford_units = afford_units.merge(skip_addresses,left_on="Address",right_on=col,how="left")
    afford_units = afford_units.loc[pd.isnull(afford_units[col]),afford_columns]
    return afford_units


def read_afford_units(coords,skip_addresses):
    afford_units = pd.read_excel("Affordable Apartments - Bond Canvassing.xls")
    afford_units["distance"] = afford_units.apply(mile_distance,coords=coords,axis=1)
    afford_units["registered_voters"] = afford_units["registered_voters"].fillna(0)
    afford_units["16_primary_first_timers"] = afford_units["16_primary_first_timers"].fillna(0)
    afford_units["16_general_voters"] = afford_units["16_general_voters"].fillna(0)
    
    afford_units["score"] = afford_units.apply(add_score,axis=1)
    afford_units["score"] += 3
    afford_units = afford_units.sort_values("score",ascending=False)
    afford_units = afford_units[afford_units["Unit_Count"]>5]
    afford_units = remove_units(afford_units,skip_addresses)

    afford_units["bigcount"] = afford_units["Unit_Count"].map(int) / 70
    afford_units["bigcount"] = afford_units["bigcount"].map(int)
    return afford_units


def split_afford_units(afford_units):
    big_units = afford_units[afford_units["bigcount"]>0].reset_index()
    small_units = afford_units[afford_units["bigcount"]==0].reset_index()
    big_units["smallcount"] = big_units["Unit_Count"]/20
    big_units["smallcount"] = big_units["smallcount"].map(int)
    small_units["smallcount"] = small_units["Unit_Count"]/10
    small_units["smallcount"] = small_units["smallcount"].map(int)
    return big_units,small_units

def main_backup_afford_units(big_afford_units,est_canvassers,percent_affordable):
    count = 0
    for ind,row in big_afford_units.iterrows():
        count += row["bigcount"]
        if count >= percent_affordable * (est_canvassers/2):
            main_afford_units = big_afford_units.loc[:ind,:]
            backup_afford_units = big_afford_units.loc[1+ind:1+2*ind]
            return main_afford_units,backup_afford_units

def get_bonus_afford_units(small_afford_units,main_afford_units):
    count = 0
    for ind,row in small_afford_units.iterrows():
        count += row["smallcount"]
        if count >= sum(main_afford_units["smallcount"]):
            return small_afford_units.loc[:ind,:]
            break

def get_market_rate_units(coords,afford_units,skip_addresses):
    skip_address_string = "'" + "','".join(skip_addresses.iloc[:,0]) + "'"

    market_query="""
    SELECT
        distinct address
        ,LAT
        ,LON
    FROM
        cutter_canvas_data ccd
    WHERE
        ccd.lat between {lat} - .05 and {lat} + .05
        and ccd.lon between {lon} -.05 and {lon} + .05
        and region = 'Austin, TX'
        and address not in ({skip_address_string})
    """.format(lat = coords[0],lon = coords[1],skip_address_string=skip_address_string)

    market_addresses = read_mysql_data(market_query)
    #print market_addresses
    

    #Remove affordable complexes
    address_join = market_addresses.merge(afford_units,how="left",left_on="address",right_on="Address")
    market_addresses = address_join.loc[pd.isnull(address_join["Address"]),("address","LAT_x","LON_x")]
    market_addresses.columns = ["address","LAT","LON"]

    market_address_string = "'" + "','".join(i for i in market_addresses["address"]) + "'"

    market_voter_query = """
    SELECT 
        address
        ,COUNT(DISTINCT VUIDNO) registered_voters
        ,SUM(CASE WHEN P16PARTY = 'DEMO' and P14PARTY = '' and P12PARTY = '' THEN 1 ELSE 0 END) 16_primary_first_timers
        ,SUM(CASE WHEN G16VOTED <> '0' THEN 1 ELSE 0 END) 16_general_voters
    FROM 
        `voter_data_Austin, TX` 
    where 
        address in ({address_list})
    GROUP BY
        address
    HAVING registered_voters >= 10
    """.format(address_list=market_address_string)  


    market_voter_counts = read_mysql_data(market_voter_query,False)
    market_rate_units = market_addresses.merge(market_voter_counts,how="inner",on="address")
    return market_rate_units

def clean_market_rate_units(market_rate_units,coords):
    market_rate_units["distance"] = market_rate_units.apply(mile_distance,coords=coords,axis=1)
    market_rate_units["registered_voters"] = market_rate_units["registered_voters"].fillna(0)
    market_rate_units["16_primary_first_timers"] = market_rate_units["16_primary_first_timers"].fillna(0)
    market_rate_units["16_general_voters"] = market_rate_units["16_general_voters"].fillna(0)
    market_rate_units["Unit_Count"] = market_rate_units["registered_voters"]
    market_rate_units["score"] = market_rate_units.apply(add_score,axis=1)
    market_rate_units = market_rate_units.sort_values("score",ascending=False)
    market_rate_units["bigcount"] = market_rate_units["Unit_Count"].map(int) / 70
    market_rate_units["bigcount"] = market_rate_units["bigcount"].map(int)
    market_rate_units["Description"] = "Market Rate"
    market_rate_units["Apartment Title"] = ""
    market_rate_units["Address"] = market_rate_units["address"]
    return market_rate_units


def split_market_rate_units(market_rate_units):
    big_market_rate = market_rate_units.loc[market_rate_units["bigcount"]>0,:].reset_index()
    small_market_rate = market_rate_units.loc[market_rate_units["bigcount"]==0,:].reset_index()
    big_market_rate["smallcount"] = big_market_rate["Unit_Count"]/20
    big_market_rate["smallcount"] = big_market_rate["smallcount"].map(int)
    small_market_rate["smallcount"] = small_market_rate["Unit_Count"]/10
    small_market_rate["smallcount"] = small_market_rate["smallcount"].map(int)
    return big_market_rate,small_market_rate


def main_backup_market_rate_units(big_market_rate_units,percent_affordable,est_canvassers):
    count = 0
    for ind,row in big_market_rate_units.iterrows():
        count += row["bigcount"]
        if count >= (1 - percent_affordable) * est_canvassers/2:
            main_market_rate_units = big_market_rate_units.loc[:ind,:]
            backup_market_rate_units = big_market_rate_units.loc[1+ind:1+2*ind]
            return main_market_rate_units,backup_market_rate_units

def get_bonus_market_rate_units(small_market_rate_units,main_market_rate_units):
    count = 0
    for ind,row in small_market_rate_units.iterrows():
        count += row["smallcount"]
        if count >= sum(main_market_rate_units["smallcount"]):
            bonus_market_rate_units = small_market_rate_units.loc[:ind,:]
            return bonus_market_rate_units


def merge_units(df1,df2):
    df1 = df1.loc[:,("Apartment Title","Address","Unit_Count","Description","registered_voters", \
                 "16_primary_first_timers","16_general_voters","LAT","LON","smallcount","bigcount","score")]
    df2 = df2.loc[:,("Apartment Title","Address","Unit_Count","Description","registered_voters", \
                     "16_primary_first_timers","16_general_voters","LAT","LON","smallcount","bigcount","score")]
    df1 = df1.append(df2).reset_index()
    return df1

def match_frames(df1,df2,colname=None):
    d1 = {}
    for j,i in df1.iterrows():
        temp_row = []
        for k,m in df2.iterrows():
            score = 0
            score += 69*((i["LAT"] - m["LAT"]) ** 2 + (i["LON"] - m["LON"])**2)**.5
            if colname:
                score += .5* abs(i[colname] - m[colname])
            temp_row.append([k,score])
        temp_row = sorted(temp_row,key = lambda x: x[1])
        #print [ind[1] for ind in temp_row]
        d1[j] = [ind[0] for ind in temp_row]
    d2 = {}
    for j,i in df2.iterrows():
        temp_row = []
        for k,m in df1.iterrows():
            score = 0
            score += 69*((i["LAT"] - m["LAT"]) ** 2 + (i["LON"] - m["LON"])**2)**.5
            if colname:
                score += .5* abs(i[colname] - m[colname])
            temp_row.append([k,score])
        temp_row = sorted(temp_row,key = lambda x: x[1])
        d2[j] = [ind[0] for ind in temp_row]
    matches = matchmaker(d2,d1)
    return matches


def bonus_match(main_units,small_units):
    big_unit_array = []
    for ind,row in main_units.iterrows():
        for n in range(row.smallcount):
            big_unit_array.append([ind,n,row.LAT,row.LON])
    big_unit_match = pd.DataFrame(big_unit_array, columns=["ind","sub_ind","LAT","LON"])
    small_unit_array = []
    for ind,row in small_units.iterrows():
        for n in range(row.smallcount):
            small_unit_array.append([ind,n,row.LAT,row.LON])
    small_unit_array = small_unit_array[:len(big_unit_array)]
    for i in range(min(0,len(big_unit_array) - len(small_unit_array))):
        small_unit_array.append([-1,-1,0,0])
    small_unit_match = pd.DataFrame(small_unit_array,columns=["ind","sub_ind","LAT","LON"])
    return big_unit_match,small_unit_match

def get_frame(row,main_units,bonus_units,big_unit_match,small_unit_match,small_dict):
    empty_frame = pd.DataFrame(columns = main_units.columns)
    filt = big_unit_match.loc[big_unit_match["ind"]==row,:]
    for ind,row in filt.iterrows():
        match_ind = small_dict[ind]
        if small_unit_match.loc[match_ind,"sub_ind"] == 0:
            small_ind = small_unit_match.loc[match_ind,"ind"]
            #print small_units.loc[small_ind,:]
            empty_frame = empty_frame.append(bonus_units.loc[small_ind,:])
    return empty_frame.reset_index(drop=True)


def apartment_query(extra=""):
    query = """
    SELECT 
        address,
        zip,
        count(distinct address_exp) units
    FROM 
        `voter_data_Austin, TX` 
    WHERE 
        1=1
        {extra}
    GROUP BY
        address,
        zip
    HAVING
        units > 10
    ORDER BY units desc
    """.format(extra=extra)
    return read_mysql_data(query)

def get_apartment_list(extra=""):
    zillow_data = ZillowWrapper("X1-ZWz1gms3bumcy3_1sm5d")
    apts = apartment_query(extra)
    full_apts = []
    for i in apts.itertuples():
        try:
            deep_search_response = zillow_data.get_deep_search_results(str(i.address),str(i.zip)[:5])
            result = GetDeepSearchResults(deep_search_response)
            if result.zestimate_amount:
                cost = float(result.zestimate_amount)
            else:
                cost = float(result.tax_value) * 1.3
            high_score = cost / 100000.0
            low_score = 100000.0 / (cost/i.units)
            if high_score > low_score:
                amt = cost/i.units
            if low_score > high_score:
                amt = cost
            full_apts.append([i.address,i.zip,i.units,amt,result.year_built])

        except Exception as e:
            print str(i.address),str(i.zip)[:5]
            print e.message
            print result
            full_apts.append([i.address,i.zip,i.units,None,None])

    #Create a final dataframe
    data = pd.DataFrame(full_apts,columns=["address","zip","units","cost","year"])
    data["city"] = "Austin"
    data["state"] = "TX"
    
    return data


def upload_apartment_list():
    data = pd.read_csv("Apartment_list.csv")
    #data["zip"] = data["zip_x"].map(str)    
    #data["units"] = data["units_x"].map(str)

    return data


def populate_missing_lat_lon(coords):
    for num,row in coords[pd.isnull(coords["LAT"])].iterrows():
        g = geocoder.arcgis(row.Address[:-11])
        latlng = g.latlng
        if not latlng:
            latlng = [None,None]
        coords.at[num,"LAT"] = latlng[0]
        coords.at[num,"LON"] = latlng[1]
        return coords

def set_apartment_score(data,center_coords,random_function,avgyear,avgcost):
    #Set the score at a default of 100
    data["score"] = 100.0
    
    #Update scores based on coordinates, year created and cost
    for ind,row in data.iterrows():
        score = 0
        score += ((center_coords[0] - row.LAT)**2 + (center_coords[1] - row.LON)**2)**.5 * 69
        score += (float(row.year) - avgyear) * (1/7)
        score += (row.cost - avgcost) * (1.6/200000.0)
        if score:
            data.at[ind,"score"] = score
            
        #Sort the data by score
    data["cost"] = data["cost"].map(int)
    data = data.sort_values("score")
    return data

#Update the lat and lng for a sheet
def update_lat_lng(lat,lon,units,add_lat,add_lon,add_units):
    try:
        new_lat = (lat * units + add_lat * add_units) / (1.0 * units + add_units)
        new_lon = (lon * units + add_lon * add_units) / (1.0 * units + add_units)
    except:
        print [lat,lon,units,add_lat,add_lon,add_units]
    return (new_lat,new_lon)


def get_distance(row,*args,**kwargs):
    lat = kwargs.pop("LAT")
    lon = kwargs.pop("LON")
    row["distance"] = ((row["LAT"] - lat) ** 2 + (row["LON"] - lon)**2)**.5
    return row



def assign_teams(data,target_teams,team_max = 75):
    team_table = pd.DataFrame([],columns=["LAT","LON","teams","big_need","small_need","units"])
    data["team"] = -1
    #Go through the list of apartments and assign them accordingly.
    for ind,row in data.iterrows():
        marker = time.clock()
        #If there are still empty teams assign this apartment to a new team
        if sum(team_table["teams"]) < target_teams:
            team_ind = len(team_table)
            if row.units >= team_max:
                teams = int(row.units)/team_max
                big_need = int(row.units)/team_max
                small_need = 4*int(row.units)/team_max
            else:
                teams = 1
                big_need = 0
                small_need = (int(team_max)/5) - int(row.units)/10
            temp_frame = pd.DataFrame([[row.LAT,row.LON,teams,big_need,small_need,row.units]],\
                                      columns=["LAT","LON","teams","big_need","small_need","units"])
            team_table = team_table.append(temp_frame,ignore_index=True)
            data.at[ind,"team"] = team_ind
        #If the teams all have at least 1 property assign it to the best property
        else:
            if row.units >= team_max:
                #print row
                count = int(row.units) / team_max
                temp_data = team_table[team_table["big_need"]>= count]
                if len(temp_data) == 0:
                    continue
                #temp_data["distance"] = ((temp_data["LAT"] - row.LAT)**2 + (temp_data["LON"] - row.LON)**2)**.5
                temp_data = temp_data.apply(get_distance,axis=1,LAT=row.LAT,LON=row.LON)
                #print temp_data
                #print team_table
                team_ind = temp_data["distance"].idxmin()
                data.at[ind,"team"] = team_ind
                team_table.at[team_ind,"big_need"] -= count
                (new_lat,new_lon) = update_lat_lng(row.LAT,row.LON,row.units,team_table.at[team_ind,"LAT"],\
                                                   team_table.at[team_ind,"LON"],team_table.at[team_ind,"units"])
                team_table.at[team_ind,"LAT"] = new_lat
                team_table.at[team_ind,"LON"] = new_lon
            else:
                count = int(row.units) / 10
                temp_data = team_table[team_table["small_need"]>= count]
                marker = time.clock()
                if len(temp_data) == 0:
                    continue
                #temp_data["distance"] = ((temp_data["LAT"] - row.LAT)**2 + (temp_data["LON"] - row.LON)**2)**.5
                temp_data = temp_data.apply(get_distance,axis=1,LAT=row.LAT,LON=row.LON)
                marker = time.clock()
                team_ind = temp_data["distance"].idxmin()
                marker = time.clock()
                data.at[ind,"team"] = team_ind
                marker = time.clock()
                team_table.at[team_ind,"small_need"] -= count
                marker = time.clock()
                (new_lat,new_lon) = update_lat_lng(row.LAT,row.LON,row.units,team_table.at[team_ind,"LAT"],\
                                                   team_table.at[team_ind,"LON"],team_table.at[team_ind,"units"])
                marker = time.clock()
                team_table.at[team_ind,"LAT"] = new_lat
                team_table.at[team_ind,"LON"] = new_lon
                marker = time.clock()
            if sum(team_table["big_need"]) == 0 and sum(team_table["small_need"])==0:
                break
    return [data,team_table]
        

def create_distance_matrix(data,team):
    temp_table = data[data["team"]==team]
    df = pd.DataFrame(columns=[str(i) for i in temp_table.index] + ['dummy'])
    for i in temp_table.itertuples():
        for j in temp_table.itertuples():
            df.loc[str(i.Index),str(j.Index)] = int(100000*((i.LAT - j.LAT)**2 + (i.LON - j.LON)**2)**.5)
        df.loc[str(i.Index),"dummy"] = 0
    df.loc['dummy',:] = [0] * len(df.columns)
    return df


# Distance callback
def create_distance_callback(df):
  #Create a callback to calculate distances between cities.

  def distance_callback(from_node, to_node):
    return df.iloc[from_node,to_node]

  return distance_callback


def get_assignment(data,team):
    distance_matrix = create_distance_matrix(data,team)
    
    tsp_size = len(distance_matrix)
    num_routes = 1
    depot = tsp_size-1
    #print tsp_size,num_routes,depot
    routing = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    # Create the distance callback.
    dist_callback = create_distance_callback(distance_matrix)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)
    
    return [assignment,routing]


def get_routes_array(assignment, num_vehicles, routing):
    routes = []
    for route_nbr in range(num_vehicles):
        node = routing.Start(route_nbr)
        route = []

        while not routing.IsEnd(node):
              index = routing.NodeToIndex(node)
              route.append(index)
              node = assignment.Value(routing.NextVar(node))
        routes.append(route)
    return routes


def get_order(route_list):
    enum_order = [[i,j] for i,j in enumerate(route_list)]
    sorted_order = sorted(enum_order,key = lambda x: x[1])
    return [i[0] for i in sorted_order]


def iterate_apts(data,center_coords,random_function,avgyear,avgcost,est_canvas_teams,team_max):
    start_time = time.clock()
    data = set_apartment_score(data,center_coords,random_function,avgyear,avgcost)
    [data,team_table] = assign_teams(data,est_canvas_teams,team_max)
    temp_routes = []
    for i in range(len(team_table)):
        [assignment,routing] = get_assignment(data,i)
        row = [get_routes_array(assignment, 1, routing)[0],assignment.ObjectiveValue()]
        temp_routes.append(row)
    return [data,team_table,temp_routes]


def random_function(num):
    return (.5 + random.random()) * num
