import json
#import glob
import os
from glob import glob

# get all csv file path from my directory and put them in a list
PATH = "C:/thesis-goettert/json/"
EXT = "*.json"
all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]	

# iterate over every single csv file
counter = 0
merged_file = {}
for i in all_csv_files:
    counter += 1
    print(counter/262428)

    # merge every file into merged_file
    with open(i, encoding="utf8") as json_file:
        data = json.load(json_file)
        merged_file.update(data)

with open('C:/thesis-goettert/merged_json/merged_file.json', 'w', encoding='utf-8') as jsonf: 
	jsonf.write(json.dumps(merged_file, indent=4))    