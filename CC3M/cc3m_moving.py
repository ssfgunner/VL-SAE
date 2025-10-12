# process the raw CC3M data into different folders based on file types

import os
import shutil
from tqdm import tqdm
from time import sleep

raw_data_dir = './cc3m_data'
json_dir = './cc3m_json'
txt_dir = './cc3m_txt'
jpg_dir = './cc3m_jpg'

if not os.path.exists(json_dir):
    os.makedirs(json_dir)
if not os.path.exists(txt_dir):
    os.makedirs(txt_dir)
if not os.path.exists(jpg_dir):   
    os.makedirs(jpg_dir)

for files in tqdm(os.listdir(raw_data_dir)):
    if files.endswith('.json'):
        shutil.move(os.path.join(raw_data_dir, files), json_dir)
    elif files.endswith('.txt'):
        shutil.move(os.path.join(raw_data_dir, files), txt_dir)
    elif files.endswith(('.png', '.jpg', '.jpeg')):
        shutil.move(os.path.join(raw_data_dir, files), jpg_dir)
    sleep(0.001)