
import os
import time
import zipfile

import gmplot
import numpy as np
from fpdf import FPDF
from pyvirtualdisplay import Display
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


#Convert a folder to a zip file
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

#Create the zip file to send out
def zip_folder(folder):
    zipf = zipfile.ZipFile(folder + '/temp_email.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir(folder + '/temp_folder', zipf)
    zipf.close()


#Take a list of addresses and make an html file with the data points plotted
def make_html_file(df,folder):
    #Find the center of addresses specified
    center_lat,center_lon = np.mean(df["LAT"]),np.mean(df["LON"])
    #Make a Google map centered on the specified address
    gmap = gmplot.GoogleMapPlotter(center_lat,center_lon,16,apikey='AIzaSyBWmNLBKBZOVA78PfJIh8H51SD0Li3b0F8')
    #Split the addresses into 4 categories
    df_no_voters = df[df["voters"]==0]
    df_some_voters = df[(df["voters"]>0) & (df["voters"]<4)]
    df_many_voters = df[(df["voters"]>3) & (df["voters"]<10)]
    df_ton_voters = df[df["voters"]>9]
    #Plot the 4 categories on the map
    gmap.scatter(df_no_voters["LAT"], df_no_voters["LON"], color="green",marker=False,size=10)
    gmap.scatter(df_some_voters["LAT"], df_some_voters["LON"], color="black",marker=False,size=8)
    gmap.scatter(df_many_voters["LAT"], df_many_voters["LON"], color="black",marker=False,size=15)
    gmap.scatter(df_ton_voters["LAT"], df_ton_voters["LON"], color="black",marker=False,size=23)
    #Save the map as an HTML file
    gmap.draw(folder + "/temp_map.html")


#Convert the html file into a png file
#This is the most time-consuming part of the process
#We are opening it in a browser and taking a screenshot
#So if there's a better way that is preferable
def make_img_file(cluster,folder):
    driver = getattr(make_img_file, 'driver', None)
    if not driver:
        #display = Display(visible=0, size=(800, 800))
        #display.start()
        #Have a virtual Chrome driver open the html file
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('headless')
        chrome_options.add_argument('no-sandbox')
        driver = webdriver.Chrome(chrome_options=chrome_options)
        #driver = webdriver(\
            #command_executor='http://172.19.0.5:4444/wd/hub',\
            #desired_capabilities=DesiredCapabilities.CHROME)
        make_img_file.driver = driver
    cwd = os.getcwd()

    # We need access to the map object to figure out when its done loading
    # however GoogleMapPlotter outputs javascript that makes it a local variable
    # so we can't get at it unless its made global.  Unfortunately, google maps
    # api has no way of getting all map objects.
    htmlpath = "{cwd}/{folder}/temp_map.html".format(cwd=cwd,folder=folder)
    with open(htmlpath) as f:
        data = f.read().replace('var map', 'window.map')
        with open(htmlpath, 'w') as f:
            f.write(data)
    
    driver.get("file://{cwd}/{folder}/temp_map.html".format(cwd=cwd,folder=folder))
    
    # Async execute this javascript, which should return when the map is
    # fully loaded
    driver.execute_async_script(r'''
// execute_async_script waits for callback to be called
var callback = arguments[arguments.length - 1];
var d = document.createElement('div');
document.getElementsByTagName('body')[0].prepend(d);
try {
    google.maps.event.addListenerOnce(map, 'tilesloaded', function (){
        d.innerHTML = '<div id="map-loaded" />';
        callback();
    });
} catch(e) {
    d.innerHTML = '<h1>Could not add loaded listener. Map may not be fully loaded!! ('+e+')</h1>';
    callback();
}
    ''')
    
    #Save the screenshot as a file
    driver.get_screenshot_as_file("{folder}/temp_map_{cluster}.png".format(cluster=cluster,folder=folder))


#Add an image to a specified pdf file
def add_img(pdf,cluster,folder,w=None):
    if w:
        pdf.image("{folder}/temp_map_{cluster}.png".format(cluster=cluster,folder=folder),w=w)
    else:
        pdf.image("{folder}/temp_map_{cluster}.png".format(cluster=cluster,folder=folder))


#For each address make cells that will be on the list of addresses given to the canvassers
#Each address gets at least a white row and a gray row
def write_address_rows(pdf,address):
    pdf.set_font('Times','',12.0) 
    th = pdf.font_size * 1.25
    pdf.cell(2.75, th, address, border=1)
    pdf.cell(1, th, "", border=1)
    pdf.cell(1, th, "", border=1)
    pdf.cell(.75, th, "", border=1)
    pdf.cell(2, th, "", border=1)
    pdf.ln(th)
    pdf.cell(2.75, th, "", border=1,fill=True)
    pdf.cell(1, th, "", border=1,fill=True)
    pdf.cell(1, th, "", border=1,fill=True)
    pdf.cell(.75, th, "", border=1,fill=True)
    pdf.cell(2, th, "", border=1,fill=True)
    pdf.ln(th)

#Write the first row on a new page - it will have the headers
def write_header_row(pdf):
    pdf.set_font('Times','',12.0) 
    th = pdf.font_size * 1.25
    pdf.cell(2.75, th, "Address", border=1)
    pdf.cell(1, th, "Unit #", border=1)
    pdf.cell(1, th, "Residence?", border=1)
    pdf.cell(.75, th, "Home?", border=1)
    pdf.cell(2, th, "Notes", border=1)
    pdf.ln(th)

#On a new page list the team #
#And add a header row
def new_page(pdf,cluster,cont=False):
    pdf.add_page()
    
    header_text = 'Turf # ' + str(cluster)
    if cont:
        header_text += " (Continued)"
    
    pdf.set_font('Times','B',16.0) 
    pdf.cell(10, 0.0, header_text, align='C')
    pdf.set_font('Times','',12.0) 
    pdf.ln(0.2)
    
    write_header_row(pdf)
    
    return pdf


#For each address write rows onto a sheet
#pdf - the pdf file
#row - a row of data represeting an address
#cluster - the number of the cluster
#row_count - how far down the sheet we are
def write_address(pdf,row,cluster,row_count):
    address = row.address
    print_doors = max(1,int(row.doors/1.5))
    #Even an address with no registered doors will get printed
    #Start at the current row on the page and add the number of doors
    for i in range(row_count,row_count + print_doors):
        #Print 2 rows for each door
        #And when we're done a page make a new page
        if row_count > 0 and row_count % 24 == 0:
            pdf = new_page(pdf,cluster,cont=True)
        write_address_rows(pdf,address)
        row_count += 1
    #Return how far down the page we are
    return row_count


#For a cluster make a new page and then write each address to that page
def write_cluster(pdf,data,cluster):
    new_page(pdf,cluster)
    filt_data = data[data["Cluster"]==cluster]
    #Sort addresses by street and number
    #This will make it easy to find the needed address
    filt_data = filt_data.sort_values(by=["STREET","NUMBER"])
    row_count = 0
    for row in filt_data.itertuples():
        row_count = write_address(pdf,row,cluster,row_count)
    #Add a blank page at the end
    #We found that when double sided printing otherwise we'd end up with cluster 2 and cluster 3 on a single page
    pdf.add_page()

#Write the sheet that we will use to assign turfs
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
    #Depending on turf size and walking distance assign a team 2 members or 4 members
    if row.doors > 1.2 * turf_size - 20 * row.walking_distance:
        pdf.cell(1.75, th, "Team Member 3", border=1)
        pdf.cell(1.75, th, "Team Member 4", border=1)
    pdf.ln(th)
    pdf.cell(1.75, th, "", border=1)
    pdf.cell(1.75, th, "", border=1)
    if row.doors > 1.6 * turf_size - 30 * row.walking_distance:
        pdf.cell(1.75, th, "", border=1)
        pdf.cell(1.75, th, "", border=1)
    pdf.ln(th)


#Add the text to a PDF page we're making
def text_page(pdf,cluster,street_list,doors,voters):
    #Add a page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 40)
    #Write the team name (based on cluster #)
    pdf.cell(30, 20, 'Team # ' + str(cluster),ln=2)
    pdf.set_font('Arial', 'B', 24)
    #Specify the # of doors and voters
    pdf.cell(20, 10, str(doors) + " registered doors",ln=2)
    pdf.cell(20, 10, str(voters) + " registered voters",ln=2)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    #Write each street in the turf
    for i in street_list:
        pdf.cell(14, 8, i,ln=2)
    #Make dots to specify what each means    
    pdf.set_fill_color(r = 0)
    pdf.ellipse(pdf.get_x() + 3,pdf.get_y() + 3,3,3,style='F')
    pdf.cell(8,3,"")
    pdf.cell(20, 10, "Address w/ voter",ln=1)
    pdf.ellipse(pdf.get_x() + 3,pdf.get_y() + 3,6,6,style='F')
    pdf.cell(12,3,"")
    pdf.cell(20, 10, "Address w/ many voters",ln=1)
    pdf.set_fill_color(r = 0, g = 255, b = 0)
    pdf.ellipse(pdf.get_x() + 3,pdf.get_y() + 3,3,3,style='F')
    pdf.cell(8,3,"")
    pdf.cell(20, 10, "Address w/ no voters",ln=1)


def make_pdf(main_units,backup_units,bonus_units,backup_dict,main_unit_match,bonus_unit_match,bonus_dict,folder_name):
    pdf = FPDF()

    for ind,row in main_units.iterrows():
        for count in range(row["bigcount"]):
            backup_ind = backup_dict[ind]



            pdf.add_page()
            pdf.set_font('Arial', 'B', 35)
            pdf.cell(0, 20, 'Team # ' + str(1+ind) +"({num} team members)".format(num=str(min(2*row["bigcount"],8))),ln=2,align='C')

            #Set the main complex
            pdf.set_font('Arial', 'B', 20)
            pdf.cell(0, 10, 'Main Complex (go here first):',ln=2)
            pdf.set_font('Arial',"", 14)

            apt_title = main_units.loc[ind,"Apartment Title"]
            description = main_units.loc[ind,"Description"]
            address = main_units.loc[ind,"Address"]


            pdf.cell(0, 7, apt_title + ",  " + address,ln=2)
            pdf.cell(0, 7, description,ln=2)
            pdf.cell(0, 12, "Check if you visited _____  Approx. what % of units did you visit ______",ln=2)

            #Set the backup complex
            pdf.set_font('Arial', 'B', 20)
            pdf.cell(0, 10, "Backup Complex (ONLY if you can't access main):",ln=2)
            pdf.set_font('Arial',"", 14)

            apt_title = backup_units.loc[backup_ind,"Apartment Title"]
            description = backup_units.loc[backup_ind,"Description"]
            address = backup_units.loc[backup_ind,"Address"]


            pdf.cell(0, 7, apt_title + ",  " + address,ln=2)
            pdf.cell(0, 7, description,ln=2)
            pdf.cell(0, 7, "Check if you visited _____  Approx. what % of units did you visit _________",ln=2)

            #Set the bonus complexes
            bonus_frame = get_frame(ind,main_units,bonus_units,main_unit_match,bonus_unit_match,bonus_dict)
            pdf.set_font('Arial', 'B', 20)
            pdf.cell(0, 10, "Bonus Complexes",ln=2)
            pdf.set_font('Arial',"", 14)

            for bonus_ind,bonus_row in bonus_frame.iterrows():
                apt_title = bonus_row["Apartment Title"]
                description = bonus_row["Description"]
                address = bonus_row["Address"]


                pdf.cell(0, 7, apt_title.encode('utf8') + ",  " + address.encode('utf8'),ln=2)
                pdf.cell(0, 7, description,ln=2)
                pdf.cell(0, 7, "Check if you visited _____  Approx. what % of units did you visit _________",ln=2)

    pdf.output(folder_name + '/temp_folder/Turf list.pdf', 'F')

def bond_assign_pdf(main_units,folder_name):
    pdf = FPDF(format='letter', unit='in',orientation='P')
    
    pdf.add_page()
    for row,temp in main_units.iterrows():

        pdf.set_font('Times','B',10.0) 
        th = pdf.font_size
        pdf.cell(1.75, th, "Team # " + str(row), border=1)
        
        pdf.set_font('Times','',10.0) 
        pdf.cell(1.75, th, str(int(temp.registered_voters)) + " registered voters", border=1)
        pdf.cell(2, th, str(temp.Address)[:24], border=1)
        pdf.cell(1.75, th, str(temp.Description), border=1)
        
        pdf.ln(th)
        pdf.cell(1.75, th, "Team Captain", border=1)
        pdf.cell(1.75, th, "Team Member 2", border=1)
        #Depending on turf size and walking distance assign a team 2 members or 4 members
        if temp["bigcount"] > 1:
            pdf.cell(2, th, "Team Member 3", border=1)
            pdf.cell(1.75, th, "Team Member 4", border=1)
        if temp["bigcount"] > 2:
            pdf.ln(th)
            pdf.cell(1.75, th, "", border=1)
            pdf.cell(1.75, th, "", border=1)
        if temp["bigcount"] > 3:
            pdf.cell(2, th, "", border=1)
            pdf.cell(1.75, th, "", border=1)
        if temp["bigcount"] > 2:
            pdf.ln(th)
            pdf.cell(1.75, th, "Team Captain 2", border=1)
            pdf.cell(1.75, th, "Team Member 6", border=1)
        if temp["bigcount"] > 3:
            pdf.cell(2, th, "Team Member 7", border=1)
            pdf.cell(1.75, th, "Team Member 8", border=1)
        pdf.ln(th)
        pdf.cell(1.75, th, "", border=1)
        pdf.cell(1.75, th, "", border=1)
        if temp["bigcount"] > 1:
            pdf.cell(2, th, "", border=1)
            pdf.cell(1.75, th, "", border=1)
        
        pdf.ln(th)
    pdf.output(folder_name + '/temp_folder/Assign Sheet.pdf', 'F')


def add_pages(temp_table,teams,pdf,ind):
    main_units = temp_table.loc[temp_table["apt_type"]=="main",:]
    backup_units = temp_table.loc[temp_table["apt_type"]=="backup",:]
    bonus_units = temp_table.loc[temp_table["apt_type"]=="bonus",:]
    for count in range(int(teams)):

        pdf.add_page()
        pdf.set_font('Arial', 'B', 35)
        pdf.cell(0, 20, 'Team # ' + str(1+ind) +"({num} team members)".format(num=str(min(2*teams,8))),ln=2,align='C')

        
        #Set the main complex
        if len(main_units) > 0:
            pdf.set_font('Arial', 'B', 20)
            pdf.cell(0, 10, 'Main Complex (go here first):',ln=2)
            pdf.set_font('Arial',"", 14)

            address = main_units.iloc[0,0]
            units = main_units.iloc[0,1]

            pdf.cell(0, 7, address + ", " + str(units) + " units",ln=2)
            pdf.cell(0, 12, "Check if you visited _____  Approx. what % of units did you visit ______",ln=2)

        #Set the backup complexes
        start = True
        for backup_ind,backup_row in backup_units.iterrows():
            if len(main_units) > 0:
                pdf.set_font('Arial', 'B', 20)
                if start:
                    pdf.cell(0, 10, "Backup Complexes (ONLY if you can't access main):",ln=2)
                    start = False
            pdf.set_font('Arial',"", 14)

            address = backup_row.address
            units = backup_row.units

            pdf.cell(0, 7, address + ", " + str(units) + " units",ln=2)
            pdf.cell(0, 7, "Check if you visited _____  Approx. what % of units did you visit _________",ln=2)

        #Set the bonus complexes
        if len(main_units) > 0:
            pdf.set_font('Arial', 'B', 20)
            pdf.cell(0, 10, "Bonus Complexes",ln=2)
        pdf.set_font('Arial',"", 14)

        for bonus_ind,bonus_row in bonus_units.iterrows():
            address = bonus_row["address"]
            units = bonus_row["units"]

            pdf.cell(0, 7, address.encode('utf8') + ", " + str(units) + " units",ln=2)

            pdf.cell(0, 7, "Check if you visited _____  Approx. what % of units did you visit _________",ln=2)

