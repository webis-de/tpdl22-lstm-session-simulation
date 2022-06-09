import json
import copy

data = json.load(open('C:/thesis-goettert/Uebergangsmatrix/uebergangsmatrix.json'))

list_of_dicts =['start', 'end', 'export_bib', 'export_cite', 'export_search_mail', 'sofis_information', 'goto_advanced_search', 'account_action', 'goto_favorites', 'goto_fulltext', 'goto_google_books', 'goto_google_scholar', 'goto_history', 'goto_home', 'goto_local_availability', 'goto_login', 'goto_topic-feeds', 'goto_topic-research', 'save_search', 'save_to_multiple_favorites', 'search', 'search_advanced', 'search_institution', 'search_keyword', 'search_person', 'view_record', 'query_form', 'docid']

row_label = {}
k = 0
while k < len(list_of_dicts):    
    row_label.update({list_of_dicts[k]:0})
    k += 1

z = 0
colummn_label = {}
while z < len(list_of_dicts):    
    colummn_label.update({list_of_dicts[z]: row_label})
    z += 1

dict_a = {}
for(k,v) in colummn_label.items():   
    for (k2,v2) in v.items():
            v[k2] = data[k + '_to_' + k2]

    dict_a[k] = copy.deepcopy(v)
with open('C:/thesis-goettert/Uebergangsmatrix/updated_matrix.json', 'w', encoding='utf-8') as fp:
    fp.write(json.dumps(dict_a, indent=4))