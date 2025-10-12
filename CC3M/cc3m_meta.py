# process the raw CC3M json files into one single json file

import os
import json
import shutil
from tqdm import tqdm
from time import sleep

json_dir = './cc3m_json'

merged_data = []
json_files = sorted(os.listdir(json_dir))

for json_file in tqdm(json_files, desc='Merging JSON files'):
    if json_file.endswith('.json'):
        with open(os.path.join(json_dir, json_file), 'r') as f:
            data = json.load(f)
            merged_data.append(data)
print(f'Merged {len(merged_data)} JSON files.')


output_file = 'merged_cc3m.json'
with open(output_file, 'w') as f:
    json.dump(merged_data, f, indent=4)
