import json
import numpy
import sys
#import itertools

f = open('C:/thesis-goettert/merged_json/merged_file2.json')
data = json.load(f)
action = 0
sort_param = []
page_param = []
informationtype_param = []
filter = []
daterange = []

for (k,v) in data.items():
    for i in v:
        for(k2,v2) in i.items():
            if(k2 == 'request'):
                if(v2 != {}):
                    for k3,v3 in v2.items():
                        if(k3 == 'sort_param'):
                            sort_param.append(v3)
                        if(k3 == 'page_param'):
                            page_param.append(v3)
                        if(k3 == 'informationtype_param'):
                            informationtype_param.append(v3)
                        if(k3 == 'filter'):
                            filter.append(v3)
                        if(k3 == 'daterange'):
                            daterange.append(v3)    

# 299064 ['score+desc', 'norm_publishDate_str+asc', 'norm_publishDate_str+desc', 'norm_date_acquisition_date+desc', 'norm_publishDate_str desc', 'norm_publishDate_str+desc+norm_date_acquisition_date+desc', 'norm_publishDate_str desc norm_date_acquisition_date desc', 'lotka', 'score desc', 'norm_publishDate_str asc', 'citation_count_int+desc', 'bradford', 'score+desc&view=rss', 'norm_date_acquisition_date desc', 'citation_count_int desc', 'abez_choyvfuQngr_fge+qrfp', 'fpber+qrfp', 'score+desc&page=2&view=list', 'score+desc&page=3&view=list', 'score+desc&page=4&view=list', 'score+desc; score+desc', 'norm_date_acquisition_date+asc,score+desc']
#print(len(sort_param), list(dict.fromkeys(sort_param)))

# 129648 some random numbers
#print(len(page_param), list(dict.fromkeys(page_param)))
#print(page_param)

# 275101 ['literature', 'project', 'project\\', 'project; project', 'literature; literature', 'literature; project']
#print(len(informationtype_param), list(dict.fromkeys(informationtype_param)))

#129648 random cut off url's
#print(len(filter), list(dict.fromkeys(filter)))

#580 random url cutoffs
#print(len(daterange), list(dict.fromkeys(daterange)))                                        