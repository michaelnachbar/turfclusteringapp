from math import radians, sin, cos, sqrt, asin
from sklearn.cluster import KMeans
import pandas as pd
import geocoder
import gmplot
import numpy as np
import csv

from fpdf import FPDF
from selenium import webdriver
import os
import time
import zipfile

import httplib2

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from apiclient import errors, discovery
import mimetypes
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase


# In[2]:

def add_distance(row,**kwargs):
    #print kwargs
    lat2 = row["LAT"]
    lon2 = row["LON"]
    distance = haversine(kwargs['lat1'],kwargs['lon1'],lat2,lon2)
    return distance


# In[3]:

def haversine(lat1, lon1, lat2, lon2):
 
  R = 6372.8 # Earth radius in kilometers
 
  dLat = radians(lat2 - lat1)
  dLon = radians(lon2 - lon1)
  lat1 = radians(lat1)
  lat2 = radians(lat2)
 
  a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
  c = 2*asin(sqrt(a))
 
  return R * c


# In[4]:

def get_coordinates(address):
    g = geocoder.google(address)
    return list(g.latlng)


# In[5]:

def make_filtered_file(data,canvas_location,turfs,turf_sizes,new_file_name):
    coords = get_coordinates(canvas_location,False)
    print coords
    lat1 = coords[0]
    lon1 = coords[1]
    data["distances"] = data.apply(add_distance,axis=1,lat1=lat1,lon1=lon1)
    data = data.sort_values("distances")
    slice_data = data[:turfs * turf_sizes]
    slice_data.to_excel(new_file_name)


# In[6]:

def get_coordinates(address,check_intersect=True):
    g = geocoder.google(address)
    if check_intersect:
        if not g.address:
            return []
        elif "&" in g.address:
            return list(g.latlng)
        else:
            return []
    return list(g.latlng)


# In[7]:

def load_threshold_dict(lon_only=False):
    intersect_data = pd.read_csv("Intersections_1.csv")
    if lon_only:
        intersect_data = intersect_data[intersect_data["Lon"].notnull()]
    threshold_dict = {}
    for row in intersect_data.itertuples():
        s1 = row.STREET1
        s2 = row.STREET2
        if not s1 in threshold_dict:
            threshold_dict[s1] = {}
        threshold_dict[s1][s2] = None
        if not s2 in threshold_dict:
            threshold_dict[s2] = {}
        threshold_dict[s2][s1] = None
    return threshold_dict


# In[8]:

def get_potential_intersections(slice_data,km_threshold = .1,lat_threshold=.0008):
    slice_data.to_excel("slice_data.xlsx")
    threshold_dict = load_threshold_dict()
    threshold_list = {}

    for i,row in enumerate(slice_data.itertuples()):
        for j,test_row in enumerate(slice_data.itertuples()):
            if j<=i:
                continue

            if row.STREET in threshold_dict and test_row.STREET in threshold_dict[row.STREET]:
                continue
            if row.STREET == test_row.STREET:
                continue
            if abs(row.LAT - test_row.LAT) > lat_threshold:
                continue
            distance = haversine(row.LAT,row.LON,test_row.LAT,test_row.LON)
            if distance > km_threshold:
                continue
            streets = tuple(sorted([row.STREET,test_row.STREET]))
            if streets not in threshold_list:
                threshold_list[streets] = distance
            elif distance < threshold_list[streets]:
                threshold_list[streets] = distance
    return threshold_list


# In[9]:

def write_row(row,new_file=False):
    if new_file:
        f = open('Intersections_1.csv','wb')
    else:
        f = open('Intersections_1.csv','ab')
    writer = csv.writer(f)
    writer.writerow(row)
    f.close()


# In[10]:

def update_thresholds(slice_data):
    threshold_list = get_potential_intersections(slice_data)
    print len(threshold_list)
    #threshold_list.to_excel("Potential.xlsx")
    x=0
    err=0
    for (i,j),k in threshold_list.iteritems():

        x+=1
        address = i + " and " + j +", Austin, TX"
        coords = get_coordinates(address)
        if coords:
            err = 0
        else:
            err +=1
        write_row([i,j,k] + coords)
        print [i,j,k] + coords
        if x > 1500:
            return False
    return True
    


# In[11]:

def update_slice_data_nulls(slice_data):
    slice_data.loc[slice_data["voters"].isnull(),"voters"] = 0
    slice_data.loc[slice_data["doors"].isnull(),"doors"] = 0
    return slice_data


# In[12]:

def update_slice_data_avgs(slice_data):
    avg_locs = pd.pivot_table(slice_data,values=("LAT","LON"),index=("STREET"),aggfunc=np.mean)
    avg_locs = avg_locs.rename_axis(None, axis=1).reset_index()
    avg_locs.columns = ["STREET","street_lat","street_lon"]
    avg_locs["street_lat"] = avg_locs["street_lat"]  
    avg_locs["street_lon"] = avg_locs["street_lon"] 
    slice_data_updated = slice_data.merge(avg_locs,on="STREET")
    return slice_data_updated


# In[13]:

def update_slice_data_clusters(slice_data_updated,num_clusters):
    fit_data = slice_data_updated.loc[:,("LAT","LON","street_lat","street_lon")]
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(fit_data)
    slice_data_updated['Cluster'] = kmeans.labels_
    return slice_data_updated


# In[14]:

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


# In[15]:

def checkstreet(street,streets,threshold_dict):
    #print street
    ret = []
    #street = "DOWNS DR"
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
        #print unchecked
        if not minkey in threshold_dict:
            #print minkey
            #print "_________________________"
            continue
        to_check = set(unchecked.keys()).intersection(threshold_dict[minkey].keys())
        for i in to_check:
            distance = minval + 1
            unchecked[i] = min(distance,unchecked[i])

    for key,val in checked.iteritems():
        if val == 999:
            ret.append(street)
            
    #print street
    #print checked

    return len(ret)


# In[16]:

def new_cluster(street,cur_cluster,slice_data_updated,threshold_dict):
    street_points = slice_data_updated.loc[(slice_data_updated["STREET"]==street) &                                 (slice_data_updated["Cluster"]==cur_cluster),
                                ("LAT","LON")]
    street_avg = np.mean(street_points["LAT"]),np.mean(street_points["LON"])
    cluster_avgs = pd.pivot_table(slice_data_updated,index="Cluster",values=("LAT","LON"),aggfunc=np.mean)
    cluster_avgs = cluster_avgs.rename_axis(None, axis=1).reset_index()
    cluster_avgs["distance"] = cluster_avgs.apply(lambda row: haversine(row["LAT"],row["LON"],street_avg[0],street_avg[1]),axis=1)
    cluster_avgs = cluster_avgs.sort_values("distance")
    #print cluster_avgs
    for count,row in enumerate(cluster_avgs.itertuples()):
        if count > 4:
            break
        if row.Cluster == cur_cluster:
            continue
        df = slice_data_updated[slice_data_updated["Cluster"]==row.Cluster]
        df = df.reset_index()
        streets = df["STREET"].unique()
        #print row.Cluster,checkstreet(street,streets,threshold_dict)
        if checkstreet(street,streets,threshold_dict) == 0:
            
            return row.Cluster
    return None


# In[17]:

def update_slice_data_check_clusters(slice_data_updated,num_clusters,threshold_dict):
    for i in range(num_clusters):
        result = check_cluster(slice_data_updated,i,threshold_dict)
        if not result:
            continue
        while result:
            bad_street = sorted(result,key=lambda k: k[1],reverse=True)[0][0]
            upd_cluster = new_cluster(bad_street,i,slice_data_updated,threshold_dict)
            if upd_cluster:
                #print bad_street,upd_cluster
                slice_data_updated.loc[(slice_data_updated.STREET==bad_street) &                                   (slice_data_updated.Cluster == i)                                   ,                                   "Cluster"] = upd_cluster
            else:
                #print bad_street,-1
                slice_data_updated.loc[(slice_data_updated.STREET==bad_street) &                                   (slice_data_updated.Cluster == i)                                   ,                                   "Cluster"] = -1
            result = check_cluster(slice_data_updated,i,threshold_dict)
    return slice_data_updated
        


# In[18]:

def check_bad_streets(slice_data_updated,threshold_dict):
    bad_streets = slice_data_updated.loc[slice_data_updated.Cluster==-1,"STREET"].unique()
    for bad_street in bad_streets:
        upd_cluster = new_cluster(bad_street,-1,slice_data_updated,threshold_dict)
        if upd_cluster:
            #print bad_street,upd_cluster
            slice_data_updated.loc[(slice_data_updated.STREET==bad_street) &                               (slice_data_updated.Cluster == -1)                               ,                               "Cluster"] = upd_cluster
    return slice_data_updated


# In[19]:

def check_split(filt_data):
    filt_data = filt_data.sort_values('doors',ascending=False).reset_index()
    max_doors = filt_data.loc[0,"doors"]
    sum_doors = sum(filt_data["doors"])
    if sum_doors - max_doors < 80:
        return False
    else:
        return True

def row_walking_distance(row):
    return haversine(row["latmin"],row["lonmin"],row["latmax"],row["lonmax"])

def get_walking_distance(row,**kwargs):
    slice_data_updated = kwargs['slice_data_updated']
    #print 'suck my dick'
    #print row['Cluster'] 
    #print int(row['Cluster'])
    #print slice_data_updated['Cluster'].head()
    df = slice_data_updated[slice_data_updated['Cluster']==int(row['Cluster'])]
    f = {"LAT": ["min","max"],"LON": ["min","max"]}
    t = df.groupby(['STREET'],as_index=False).agg(f)
    t.columns = ["STREET","latmin","latmax","lonmin","lonmax"]
    try:
        return sum(t.apply(row_walking_distance,axis=1))
    except:
        print 'error'
        print t
        return 0

def row_distance(row,**kwargs):
        center_lat = kwargs['center_lat']
        center_lon = kwargs['center_lon']
        #try:
        return haversine(row["LAT"],row["LON"],center_lat,center_lon)

def update_cluster_numbers(slice_data_updated):
    eligible_data = slice_data_updated[slice_data_updated["Cluster"]!=-1]
    cluster_avgs = eligible_data.groupby("Cluster",as_index=False)["LAT","LON"].mean()
    [center_lat,center_lon] = eligible_data.loc[0,"LAT"],eligible_data.loc[0,"LON"]
    cluster_avgs["distance"] = cluster_avgs.apply(row_distance,axis=1,center_lat=center_lat,center_lon=center_lon)
    cluster_avgs = cluster_avgs.sort_values("distance")
    cluster_avgs = cluster_avgs.reset_index(drop=True)
    cluster_avgs = cluster_avgs.reset_index()
    merge_data = cluster_avgs.loc[:,("index","Cluster")]
    slice_data_updated = slice_data_updated.merge(merge_data,on="Cluster",how="left")
    slice_data_updated = slice_data_updated.drop('Cluster',1)
    slice_data_updated.columns = list(slice_data_updated.columns[:-1]) + ['Cluster']
    slice_data_updated.loc[pd.isnull(slice_data_updated["Cluster"]),"Cluster"] = -1
    slice_data_updated["Cluster"] = slice_data_updated["Cluster"].astype(int)
    return slice_data_updated


def get_cluster_totals(slice_data_updated):
    eligible_data = slice_data_updated[slice_data_updated["Cluster"]!=-1]
    f = {"voters": ['sum',len],"doors": 'sum', "LAT": 'mean',"LON": "mean"}
    cluster_totals = eligible_data.groupby("Cluster",as_index=False).agg(f)
    cluster_totals["Cluster"] = cluster_totals["Cluster"].astype(int)
    [center_lat,center_lon] = slice_data_updated.loc[0,"LAT"],slice_data_updated.loc[0,"LON"]
    cluster_totals["distance"] = cluster_totals.apply(row_distance,axis=1,center_lat=center_lat,center_lon=center_lon)
    cluster_totals["walking_distance"] = cluster_totals.apply(get_walking_distance,slice_data_updated=slice_data_updated,axis=1)
    cluster_totals.columns = ["Cluster","latmean","lonmean","doors","voters","addresses","distance","walking_distance"]
    return cluster_totals

# In[22]:

def split_cluster(df,cluster,splits):
    df = df.copy()
    filt_data = df.loc[df["Cluster"]==cluster,("LAT","LON","street_lat","street_lon","Cluster","doors")].copy()
    if not check_split(filt_data):
        return df
    max_cluster = max(df["Cluster"])
    print "Cluster:" + str(cluster)
    print "# of clusters:" + str(max_cluster)
    print "# of doors:" + str(sum(filt_data["doors"]))
    print "# of records: " + str(len(filt_data))
    print "Cluster Totals Says: "
    #print cluster_totals.loc[cluster_totals["index"]==cluster,:]
    #print filt_data
    kmeans = KMeans(n_clusters=splits, random_state=10).fit(filt_data)
    filt_data["label"] = kmeans.labels_
    for i in range(1,splits):
        filt_data.loc[filt_data["label"]==i,"Cluster"] = max_cluster + i
    df.update(filt_data["Cluster"])
    #print df.head()
    print "Cluster 1 len: " + str(len(df[df["Cluster"]==cluster]))
    print "Max cluster len: " + str(len(df[df["Cluster"]==max_cluster+1]))
    return df


# In[23]:

def top_street(slice_data_updated,cluster):
    filt_data = slice_data_updated.loc[slice_data_updated["Cluster"]==cluster,:]
    agg_data = filt_data.groupby("STREET").agg(len).reset_index()
    street = agg_data.loc[0,"STREET"]
    return street


# In[24]:

def new_whole_cluster(cluster_totals,slice_data_updated,cluster,threshold_dict,turf_size,missing_clusters):
    cluster_doors = cluster_totals.loc[cluster_totals["Cluster"]==cluster,"doors"]
    cluster_doors = cluster_doors.iloc[0]
    cluster_walking_distance = cluster_totals.loc[cluster_totals["Cluster"]==cluster,"walking_distance"]
    cluster_walking_distance = cluster_walking_distance.iloc[0]
    cluster_lat = cluster_totals.loc[cluster_totals["Cluster"]==cluster,"latmean"].iloc[0]
    cluster_lon = cluster_totals.loc[cluster_totals["Cluster"]==cluster,"lonmean"].iloc[0]
    street_avg = cluster_lat,cluster_lon
    cluster_avgs = pd.pivot_table(slice_data_updated,index="Cluster",values=("LAT","LON"),aggfunc=np.mean)
    cluster_avgs = cluster_avgs.rename_axis(None, axis=1).reset_index()
    cluster_avgs["distance"] = cluster_avgs.apply(lambda row: haversine(row["LAT"],row["LON"],street_avg[0],street_avg[1]),axis=1)
    cluster_avgs = cluster_avgs.sort_values("distance")
    
    street = top_street(slice_data_updated,cluster)
    #print cluster_avgs
    for count,row in enumerate(cluster_avgs.itertuples()):
        if count > 4:
            break
        if row.Cluster in missing_clusters:
            break
        if row.Cluster == cluster:
            continue
        df = slice_data_updated[slice_data_updated["Cluster"]==row.Cluster]
        df_doors = sum(df["doors"])
        try:
            walking_distance = cluster_totals.loc[cluster_totals["Cluster"]==row.Cluster,"walking_distance"]
            #walking_distance = walking_distance.reset_index()
            walking_distance = walking_distance.iloc[0]
        except:
            print 'no walking distance'
            print walking_distance
            print type(walking_distance)
            print row.Cluster
            walking_distance = 0
        
        max_doors = 2.4 * turf_size - 30 * walking_distance - 30 * cluster_walking_distance
        if df_doors + cluster_doors >= max_doors:
            return None
        df = df.reset_index()
        streets = df["STREET"].unique()
        #print row.Cluster,checkstreet(street,streets,threshold_dict)
        if checkstreet(street,streets,threshold_dict) == 0:
            return row.Cluster
    return None


# In[14]:

def make_file(df):
    center_lat,center_lon = np.mean(df["LAT"]),np.mean(df["LON"])
    print center_lat,center_lon
    gmap = gmplot.GoogleMapPlotter(center_lat,center_lon,16)
    
    df_no_voters = df[df["voters"]==0]
    df_some_voters = df[(df["voters"]>0) & (df["voters"]<4)]
    df_many_voters = df[(df["voters"]>3) & (df["voters"]<10)]
    df_ton_voters = df[df["voters"]>9]
    gmap.scatter(df_no_voters["LAT"], df_no_voters["LON"], color="yellow",marker=False,size=10)
    gmap.scatter(df_some_voters["LAT"], df_some_voters["LON"], color="black",marker=False,size=8)
    gmap.scatter(df_many_voters["LAT"], df_many_voters["LON"], color="black",marker=False,size=15)
    gmap.scatter(df_ton_voters["LAT"], df_ton_voters["LON"], color="black",marker=False,size=23)
    gmap.draw("temp_map.html")


# In[15]:

def text_page(pdf,cluster,street_list,doors,voters):
    pdf.add_page()
    pdf.set_font('Arial', 'B', 50)
    pdf.cell(30, 20, 'Team # ' + str(cluster),ln=2)
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(20, 10, str(doors) + " registered doors",ln=2)
    pdf.cell(20, 10, str(voters) + " registered voters",ln=2)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    for i in street_list:
        pdf.cell(16, 8, i,ln=2)
        
    pdf.set_fill_color(r = 0)
    pdf.ellipse(pdf.get_x() + 3,pdf.get_y() + 3,3,3,style='F')
    pdf.cell(8,3,"")
    pdf.cell(20, 10, "Address w/ voter",ln=1)
    pdf.ellipse(pdf.get_x() + 3,pdf.get_y() + 3,6,6,style='F')
    pdf.cell(12,3,"")
    pdf.cell(20, 10, "Address w/ many voters",ln=1)
    pdf.set_fill_color(r = 255, g = 255, b = 0)
    pdf.ellipse(pdf.get_x() + 3,pdf.get_y() + 3,3,3,style='F')
    pdf.cell(8,3,"")
    pdf.cell(20, 10, "Address w/ no voters",ln=1)


# In[16]:

def get_street_list(df):
    streets = pd.pivot_table(df,index="STREET",values="NUMBER",aggfunc=(min,max))
    streets = streets.rename_axis(None, axis=1).reset_index()
    ret = []
    for row in streets.itertuples():
        ret.append(str(row.min) + " " + row.STREET + " to " + str(row.max) + " " + row.STREET)
    return ret


# In[17]:

def make_img_file(cluster):
    driver = webdriver.Chrome()  # Optional argument, if not specified will search path.
    driver.get("file:///home/mike/canvas_cutting/canvas_cutting/temp_map.html")
    time.sleep(4)      
    driver.get_screenshot_as_file("temp_map_{cluster}.png".format(cluster=cluster))
    driver.quit()


# In[18]:

def add_img(pdf,cluster,w=None):
    #pdf.add_page()
    if w:
        pdf.image("temp_map_{cluster}.png".format(cluster=cluster),w=w)
    else:
        pdf.image("temp_map_{cluster}.png".format(cluster=cluster))


def num_rows(row):
    return min(2,2*int(row.doors)/2)



def write_address_rows(pdf,address):
    pdf.set_font('Times','',13.0) 
    th = pdf.font_size
    pdf.cell(4, th, address, border=1)
    pdf.cell(1.25, th, "", border=1)
    pdf.cell(1, th, "", border=1)
    pdf.cell(.75, th, "", border=1)
    pdf.cell(3, th, "", border=1)
    pdf.ln(th)
    pdf.cell(4, th, "", border=1,fill=True)
    pdf.cell(1.25, th, "", border=1,fill=True)
    pdf.cell(1, th, "", border=1,fill=True)
    pdf.cell(.75, th, "", border=1,fill=True)
    pdf.cell(3, th, "", border=1,fill=True)
    pdf.ln(th)


def write_header_row(pdf):
    pdf.set_font('Times','',13.0) 
    th = pdf.font_size
    pdf.cell(4, th, "Address", border=1)
    pdf.cell(1.25, th, "Unit #", border=1)
    pdf.cell(1, th, "Residence?", border=1)
    pdf.cell(.75, th, "Home?", border=1)
    pdf.cell(3, th, "Notes", border=1)
    pdf.ln(th)


def new_page(pdf,cluster,cont=False):
    pdf.add_page()
    
    header_text = 'Turf # ' + str(cluster)
    if cont:
        header_text += " (Continued)"
    
    pdf.set_font('Times','B',14.0) 
    pdf.cell(10, 0.0, header_text, align='C')
    pdf.set_font('Times','',13.0) 
    pdf.ln(0.5)
    
    write_header_row(pdf)
    
    return pdf



def write_address(pdf,row,cluster,row_count):
    address = row.address
    print_doors = max(1,row.doors)
    for i in range(row_count,row_count + print_doors):
        if row_count > 0 and row_count % 18 == 0:
            pdf = new_page(pdf,cluster,cont=True)
        write_address_rows(pdf,address)
        row_count += 1
    return row_count


def write_cluster(pdf,data,cluster):
    new_page(pdf,cluster)
    filt_data = data[data["Cluster"]==cluster]
    filt_data = filt_data.sort_values(by=["STREET","NUMBER"])
    row_count = cluster
    for row in filt_data.itertuples():
        row_count = write_address(pdf,row,cluster,row_count)


def write_assign_sheet(pdf,row,turf_size):
    pdf.set_font('Times','B',10.0) 
    th = pdf.font_size
    pdf.cell(1.75, th, "Team # " + str(row.Cluster), border=1)
    pdf.set_font('Times','',10.0) 
    pdf.cell(1.75, th, str(row.doors) + " registered doors", border=1)
    pdf.cell(1.75, th, str(row.voters) + " registered voters", border=1)
    pdf.cell(1.75, th, str(row.walking_distance) + " km", border=1)
    pdf.ln(th)
    pdf.cell(1.75, th, "Team Captain", border=1)
    pdf.cell(1.75, th, "Team Member 2", border=1)
    if row.doors > 1.6 * turf_size - 30 * row.walking_distance:
        pdf.cell(1.75, th, "Team Member 3", border=1)
        pdf.cell(1.75, th, "Team Member 4", border=1)
    pdf.ln(th)
    pdf.cell(1.75, th, "", border=1)
    pdf.cell(1.75, th, "", border=1)
    if row.doors > 1.6 * turf_size - 30 * row.walking_distance:
        pdf.cell(1.75, th, "", border=1)
        pdf.cell(1.75, th, "", border=1)
    pdf.ln(th)


def get_credentials():
    flags = None
    SCOPES = 'https://www.googleapis.com/auth/gmail.send'
    CLIENT_SECRET_FILE = 'client_secret.json'
    APPLICATION_NAME = 'Turf Clustering App'
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'gmail-python-email-send.json')
    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        flags = tools.argparser.parse_args(args=[])
        credentials = tools.run_flow(flow, store,flags)
        print('Storing credentials to ' + credential_path)
    return credentials



def SendMessage(sender, to, subject, msgHtml, msgPlain, attachmentFile=None):
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('gmail', 'v1', http=http)
    if attachmentFile:
        message1 = createMessageWithAttachment(sender, to, subject, msgHtml, msgPlain, attachmentFile)
    else: 
        message1 = CreateMessageHtml(sender, to, subject, msgHtml, msgPlain)
    result = SendMessageInternal(service, "me", message1)
    return result

def SendMessageInternal(service, user_id, message):
    try:
        message = (service.users().messages().send(userId=user_id, body=message).execute())
        print('Message Id: %s' % message['id'])
        return message
    except errors.HttpError as error:
        print('An error occurred: %s' % error)
        return "Error"
    return "OK"

def CreateMessageHtml(sender, to, subject, msgHtml, msgPlain):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = to
    msg.attach(MIMEText(msgPlain, 'plain'))
    msg.attach(MIMEText(msgHtml, 'html'))
    return {'raw': base64.urlsafe_b64encode(msg.as_string())}

def createMessageWithAttachment(
    sender, to, subject, msgHtml, msgPlain, attachmentFile):
    """Create a message for an email.

    Args:
      sender: Email address of the sender.
      to: Email address of the receiver.
      subject: The subject of the email message.
      msgHtml: Html message to be sent
      msgPlain: Alternative plain text message for older email clients          
      attachmentFile: The path to the file to be attached.

    Returns:
      An object containing a base64url encoded email object.
    """
    message = MIMEMultipart('mixed')
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject

    messageA = MIMEMultipart('alternative')
    messageR = MIMEMultipart('related')

    messageR.attach(MIMEText(msgHtml, 'html'))
    messageA.attach(MIMEText(msgPlain, 'plain'))
    messageA.attach(messageR)

    message.attach(messageA)

    print("create_message_with_attachment: file: %s" % attachmentFile)
    content_type, encoding = mimetypes.guess_type(attachmentFile)

    if content_type is None or encoding is not None:
        content_type = 'application/octet-stream'
    main_type, sub_type = content_type.split('/', 1)
    if main_type == 'text':
        fp = open(attachmentFile, 'rb')
        msg = MIMEText(fp.read(), _subtype=sub_type)
        fp.close()
    elif main_type == 'image':
        fp = open(attachmentFile, 'rb')
        msg = MIMEImage(fp.read(), _subtype=sub_type)
        fp.close()
    elif main_type == 'audio':
        fp = open(attachmentFile, 'rb')
        msg = MIMEAudio(fp.read(), _subtype=sub_type)
        fp.close()
    else:
        fp = open(attachmentFile, 'rb')
        msg = MIMEBase(main_type, sub_type)
        msg.set_payload(fp.read())
        fp.close()
    filename = os.path.basename(attachmentFile)
    msg.add_header('Content-Disposition', 'attachment', filename=filename)
    message.attach(msg)

    return {'raw': base64.urlsafe_b64encode(message.as_string())}

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def zip_folder(folder):
    zipf = zipfile.ZipFile('temp_email.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir('temp_folder/', zipf)
    zipf.close()

def send_email(to):
    zip_folder('temp_folder')
    SendMessage("turfclusteringapp@gmail.com", to, "Here are your turfs", "", "Here is some stuff", attachmentFile='temp_email.zip')

def send_error_email(to):
    SendMessage("turfclusteringapp@gmail.com", to, "Error making your turfs", "", "We were not able to generate your turfs. Unfortunately we hit the API limit checking addresses to make your turfs. Please run again in 24 hours.")
