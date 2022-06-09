import json

data = json.load(open('C:/thesis-goettert/merged_json/merged_file.json'))

number_of_keys = (len(data.keys()))
action_CTS_search = 0

matrix_values = {}
k = 0
list_of_keynames =['start', 'end', 'delete_comment', 'export_bib', 'export_cite', 'export_search_mail', 'sofis_information', 'goto_advanced_search', 'account_action', 'goto_favorites', 'goto_fulltext', 'goto_google_books', 'goto_google_scholar', 'goto_history', 'goto_home', 'goto_local_availability', 'goto_login', 'goto_thesaurus', 'goto_topic-feeds', 'goto_topic-research', 'save_search', 'save_to_multiple_favorites', 'search', 'search_advanced', 'search_institution', 'search_keyword', 'search_person', 'view_record', 'query_form', 'docid']
while k < len(list_of_keynames):
    j = 0
    while j < len(list_of_keynames):
        key = list_of_keynames[j] + '_to_' + list_of_keynames[k]
        matrix_values[key] = 0 
        j += 1
    k += 1

count_keys = 0
count_to_end = 0
count_ends = 0
sofis_information = ['goto_about','goto_contribute', 'goto_impressum','goto_partner','goto_sofis','goto_team']
account_action = ['goto_create_account','goto_delete_account','goto_edit_password']
for(k,v) in data.items():
    print(count_keys/number_of_keys)
    count_keys += 1
    for i in v:
        count_to_end += 1 
        origin_action = i['origin_action']
        action = i['action']
        if(origin_action in sofis_information):
            origin_action = 'sofis_information'
        if(origin_action in account_action):
            origin_action = 'account_action'
        if(action in sofis_information):
            action = 'sofis_information'    
        if(action in account_action):
            action = 'account_action'     
        if(count_to_end == len(v)):
            count_ends += 1
            matrix_values[action + '_to_end'] += 1
            count_to_end = 0   
        else:
            if(origin_action == ''):
                matrix_values['start_to_' + action] += 1  
            else:    
                matrix_values[origin_action + '_to_' + action] += 1    
print(count_ends)        
with open('C:/thesis-goettert/Uebergangsmatrix/uebergangsmatrix.json', 'w', encoding='utf-8') as fp:
    fp.write(json.dumps(matrix_values, indent=4))