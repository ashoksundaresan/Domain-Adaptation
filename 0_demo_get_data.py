
import requests
import json
import pandas as pd
import os
import paramiko
from tqdm import *


# Get cameras and gateway parameters
# url = 'http://192.168.66.116:9090/image_paths'
ip='192.168.66.116'
endpoint =':9090/image_paths'
username='ubuntu'
password='ubuntu'
url='http://'+ip+endpoint

# Set serving directory
serving_dir='/Users/karthikkappaganthu/Documents/online_learning/demo_testing_data/0_cam_imgs/cam1_mod1'
cam_id=1
model_id=1

# Setup serving dir
os.system('mkdir '+serving_dir)
os.system('mkdir '+serving_dir+'/IMG')

# Get image list
print('Getting image list from gateway ...')
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
payload = {'camera_id':cam_id, 'model_id':model_id}
resp = requests.post(url, json.dumps(payload), headers=headers)
inp=json.dumps(resp.json())
img_names_df=pd.read_json(inp)
print('Retrieved images')

# Establish connect to gateway
print('Setting up connection to gateway to download images ...')
try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, username=username,password=password)
    sftp=ssh.open_sftp()
except:
    print('Unable to connect to gateway')
    raise Exception
print('Conneted to gateway')
# Get Images
downloaded_imgs=[]
num_imgs=0


for img in tqdm(img_names_df['image_path_list']):
    img_name=img.split('/')[-1]

    if img_name.split('.')[-1]in ['jpg','jpeg','JPEG']:
        img_save_name=serving_dir+'/IMG/'+img_name
        sftp.get(img,img_save_name)
        num_imgs+=1
        downloaded_imgs.append(img_save_name)
ssh.close()

print('Download {0} images'.format(num_imgs))
downloaded_imgs_df=pd.DataFrame(downloaded_imgs)
downloaded_imgs_df.to_csv(serving_dir+'/target_data_cam_id{0}_model_id{1}.csv'.format(cam_id,model_id),index=None,sep=' ',header=None)