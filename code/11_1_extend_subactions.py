import json
import numpy
import sys
#import itertools

f = open('C:/thesis-goettert/merged_json/merged_file.json')
data = json.load(f)
action = 0

for (k,v) in data.items():
    for i in v:
        for(k2,v2) in i.items():
            if(k2 == 'action'):
                action = v2
            if(k2 == 'action_substate'):
                if(v2 == []):
                    v2.append(action)

             

with open('C:/thesis-goettert/merged_json/merged_file2.json', 'w', encoding='utf-8') as jsonf: 
	jsonf.write(json.dumps(data, indent=4))                  