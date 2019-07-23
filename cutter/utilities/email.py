from __future__ import absolute_import

import os
import smtplib
import email
import mimetypes

from email.MIMEMultipart import MIMEMultipart
from email.Utils import COMMASPACE
from email.MIMEBase import MIMEBase
from email.parser import Parser
from email.MIMEImage import MIMEImage
from email.MIMEText import MIMEText
from email.MIMEAudio import MIMEAudio
from email.encoders import encode_base64

from .output import zip_folder

__all__ = ['send_smtp_email', 'send_email', 'send_file_email', 'send_nofile_email', 'send_error_email', 'send_error_report']


#Send email
def send_smtp_email(to,address,subject,body,pw,attach=None):
    smtp_host = 'smtp.gmail.com'
    smtp_port = 587
    server = smtplib.SMTP()
    server.connect(smtp_host,smtp_port)
    server.ehlo()
    server.starttls()
    server.login(address,pw)
    
    msg = email.MIMEMultipart.MIMEMultipart()
    msg['From'] = address
    msg['To'] = to
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    if attach:
        attachments = [attach]
        for filename in attachments:
            mimetype, encoding = mimetypes.guess_type(filename)
            mimetype = mimetype.split('/', 1)
            fp = open(filename, 'rb')
            attachment = MIMEBase(mimetype[0], mimetype[1])
            attachment.set_payload(fp.read())
            fp.close()
            encode_base64(attachment)
            attachment.add_header('Content-Disposition', 'attachment',
                                  filename=os.path.basename(filename))
            msg.attach(attachment)
    server.sendmail(address,to,msg.as_string())

#Send the email
def send_email(to,folder_path):
    zip_folder(folder_path)
    send_smtp_email(to,"turfclusteringapp@gmail.com","Here are your turfs","Here's some stuff","bagelsandwiches",attach=folder_path + '/temp_email.zip')

def send_file_email(to,file_path,email_text):
    send_smtp_email(to,"turfclusteringapp@gmail.com","Here is your file",email_text,"bagelsandwiches",attach=file_path)


def send_nofile_email(to,email_text):
    send_smtp_email(to,"turfclusteringapp@gmail.com","Here is your file",email_text,"bagelsandwiches")


#Send an error email if a process fails
def send_error_email(to):
    send_smtp_email(to,"turfclusteringapp@gmail.com","Error making your turfs","We were not able to generate your turfs. Unfortunately we hit the API limit checking addresses to make your turfs. Please run again in 24 hours.","bagelsandwiches")

#Send an error email if a process fails
def send_error_report(to,e):
    import traceback
    send_smtp_email(to + ",michael.l.nachbar@gmail.com","turfclusteringapp@gmail.com","Error making your turfs","Sorry your report hit an error. Info is below. For questions reach out to michael.l.nachbar@gmail.com" + "\n" + "\n" + str(e),"bagelsandwiches")

