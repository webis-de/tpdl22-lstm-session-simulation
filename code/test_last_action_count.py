import json
import numpy
import sys
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
#import itertools

f = open('C:/thesis-goettert/merged_json/merged_file2.json')
data = json.load(f)
end_of_session_list = []

for (k,v) in data.items():
    for i in v:
        for (k2,v2) in v[len(v)-1].items():
            if(k2 == 'action'):
                end_of_session_list.append(v2)
        

#print(list(dict.fromkeys(end_of_session_list))) 
#print(len(list(dict.fromkeys(end_of_session_list))))   
a = Counter(end_of_session_list)   
a = a.most_common() 

labels = [x[0] for x in a]
values = [x[1] for x in a]
print(labels)
print(values)

indexes = np.arange(len(labels))
width = 1

plt.figure(figsize=(11,7))
plt.bar(indexes, values, width)
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.5)
plt.xticks(indexes, labels)
plt.show()
