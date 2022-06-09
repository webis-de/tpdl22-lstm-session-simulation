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
    end_of_session_list.append(len(v))

end_of_session_list_small = [x for x in end_of_session_list if x <= 25]
end_of_session_list_small_length = len(end_of_session_list_small)

end_of_session_list = Counter(end_of_session_list)   
end_of_session_list = end_of_session_list.most_common() 

end_of_session_list_small = Counter(end_of_session_list_small)  
end_of_session_list_small = {k: v / end_of_session_list_small_length for k, v in end_of_session_list_small.items()} 

print(end_of_session_list)
print(end_of_session_list_small)