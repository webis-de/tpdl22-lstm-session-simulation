import json
import numpy
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
#import itertools

f = open('C:/thesis-goettert/merged_json/merged_file2.json')
data = json.load(f)

actionlist = ['view_record', 'search_person', 'search', 'goto_google_books', 'goto_google_scholar', 'export_cite', 'goto_home', 'query_form', 'goto_local_availability', 'search_institution', 'goto_login', 'goto_favorites', 'docid', 'goto_fulltext', 'goto_advanced_search', 'search_advanced', 'goto_thesaurus', 'goto_sofis', 'goto_history', 'save_search', 'search_keyword', 'goto_about', 'goto_topic-feeds', 'goto_topic-research', 'goto_create_account', 'export_search_mail', 'export_bib', 'save_to_multiple_favorites', 'goto_partner', 
'goto_impressum', 'goto_team', 'goto_contribute', 'delete_comment', 'goto_edit_password', 'goto_delete_account']

view_record = []
search_person = []
search = []
goto_google_books = []
goto_google_scholar = []
export_cite = []
goto_home = []
query_form = []
goto_local_availability = []
search_institution = []
goto_login = []
goto_favorites = []
docid = []
goto_fulltext = []
goto_advanced_search = []
search_advanced = []
goto_thesaurus = []
goto_sofis = []
goto_history = []
save_search = []
search_keyword = []
goto_about = []
goto_topic_feeds = []
goto_topic_research = []
goto_create_account = []
export_search_mail = []
export_bib = []
save_to_multiple_favorites = []
goto_partner = []
goto_impressum = []
goto_team = []
goto_contribute = []
delete_comment = []
goto_edit_password = []
goto_delete_account = []

for (k,v) in data.items():
    for i in v:
        for(k2,v2) in i.items():
            if(k2 == 'action'):   
                    action = v2
            if(k2 == 'action_length'):
                length = v2       
        if (action == 'view_record'):
            view_record.append(length)
        elif (action == 'search_person'):
            search_person.append(length)    
        elif (action == 'search'):
            search.append(length)
        elif (action == 'goto_google_books'):
            goto_google_books.append(length)
        elif (action == 'goto_google_scholar'):
            goto_google_scholar.append(length)
        elif (action == 'export_cite'):
            export_cite.append(length)
        elif (action == 'goto_home'):
            goto_home.append(length)
        elif (action == 'query_form'):
            query_form.append(length)
        elif (action == 'goto_local_availability'):
            goto_local_availability.append(length)
        elif (action == 'search_institution'):
            search_institution.append(length)
        elif (action == 'goto_login'):
            goto_login.append(length)
        elif (action == 'goto_favorites'):
            goto_favorites.append(length)
        elif (action == 'docid'):
            docid.append(length)
        elif (action == 'goto_fulltext'):
            goto_fulltext.append(length)
        elif (action == 'goto_advanced_search'):
            goto_advanced_search.append(length)
        elif (action == 'search_advanced'):
            search_advanced.append(length)
        elif (action == 'goto_thesaurus'):
            goto_thesaurus.append(length)
        elif (action == 'goto_sofis'):
            goto_sofis.append(length)                                                                
        elif (action == 'goto_history'):
            goto_history.append(length)
        elif (action == 'save_search'):
            save_search.append(length)
        elif (action == 'search_keyword'):
            search_keyword.append(length)
        elif (action == 'goto_about'):
            goto_about.append(length)
        elif (action == 'goto_topic-feeds'):
            goto_topic_feeds.append(length)
        elif (action == 'goto_topic-research'):
            goto_topic_research.append(length)
        elif (action == 'goto_create_account'):
            goto_create_account.append(length)
        elif (action == 'export_search_mail'):
            export_search_mail.append(length)
        elif (action == 'export_bib'):
            export_bib.append(length)
        elif (action == 'save_to_multiple_favorites'):
            save_to_multiple_favorites.append(length)
        elif (action == 'goto_partner'):
            goto_partner.append(length)
        elif (action == 'goto_impressum'):
            goto_impressum.append(length)    
        elif (action == 'goto_team'):
            goto_team.append(length)
        elif (action == 'goto_contribute'):
            goto_contribute.append(length)
        elif (action == 'delete_comment'):
            delete_comment.append(length)
        elif (action == 'goto_edit_password'):
            goto_edit_password.append(length)
        elif (action == 'goto_delete_account'):
            goto_delete_account.append(length)         

for (k,v) in data.items():
    for i in v[len(v)-1]:
        if (v[len(v)-1]['action'] == 'view_record'):
            v[len(v)-1]['action_length'] = random.choice(view_record)
        elif (v[len(v)-1]['action'] == 'search_person'):
            v[len(v)-1]['action_length'] = random.choice(search_person)    
        elif (v[len(v)-1]['action'] == 'search'):
            v[len(v)-1]['action_length'] = random.choice(search)
        elif (v[len(v)-1]['action'] == 'goto_google_books'):
            v[len(v)-1]['action_length'] = random.choice(goto_google_books)
        elif (v[len(v)-1]['action'] == 'goto_google_scholar'):
            v[len(v)-1]['action_length'] = random.choice(goto_google_scholar)
        elif (v[len(v)-1]['action'] == 'export_cite'):
            v[len(v)-1]['action_length'] = random.choice(export_cite)
        elif (v[len(v)-1]['action'] == 'goto_home'):
            v[len(v)-1]['action_length'] = random.choice(goto_home)
        elif (v[len(v)-1]['action'] == 'query_form'):
            v[len(v)-1]['action_length'] = random.choice(query_form)
        elif (v[len(v)-1]['action'] == 'goto_local_availability'):
            v[len(v)-1]['action_length'] = random.choice(goto_local_availability)
        elif (v[len(v)-1]['action'] == 'search_institution'):
            v[len(v)-1]['action_length'] = random.choice(search_institution)
        elif (v[len(v)-1]['action'] == 'goto_login'):
            v[len(v)-1]['action_length'] = random.choice(goto_login)
        elif (v[len(v)-1]['action'] == 'goto_favorites'):
            v[len(v)-1]['action_length'] = random.choice(goto_favorites)
        elif (v[len(v)-1]['action'] == 'docid'):
            v[len(v)-1]['action_length'] = random.choice(docid)
        elif (v[len(v)-1]['action'] == 'goto_fulltext'):
            v[len(v)-1]['action_length'] = random.choice(goto_fulltext)
        elif (v[len(v)-1]['action'] == 'goto_advanced_search'):
            v[len(v)-1]['action_length'] = random.choice(goto_advanced_search)
        elif (v[len(v)-1]['action'] == 'search_advanced'):
            v[len(v)-1]['action_length'] = random.choice(search_advanced)
        elif (v[len(v)-1]['action'] == 'goto_thesaurus'):
            v[len(v)-1]['action_length'] = random.choice(goto_thesaurus)
        elif (v[len(v)-1]['action'] == 'goto_sofis'):
            v[len(v)-1]['action_length'] = random.choice(goto_sofis)                                                                
        elif (v[len(v)-1]['action'] == 'goto_history'):
            v[len(v)-1]['action_length'] = random.choice(goto_history)
        elif (v[len(v)-1]['action'] == 'save_search'):
            v[len(v)-1]['action_length'] = random.choice(save_search)
        elif (v[len(v)-1]['action'] == 'search_keyword'):
            v[len(v)-1]['action_length'] = random.choice(search_keyword)
        elif (v[len(v)-1]['action'] == 'goto_about'):
            v[len(v)-1]['action_length'] = random.choice(goto_about)
        elif (v[len(v)-1]['action'] == 'goto_topic-feeds'):
            v[len(v)-1]['action_length'] = random.choice(goto_topic_feeds)
        elif (v[len(v)-1]['action'] == 'goto_topic-research'):
            v[len(v)-1]['action_length'] = random.choice(goto_topic_research)
        elif (v[len(v)-1]['action'] == 'goto_create_account'):
            v[len(v)-1]['action_length'] = random.choice(goto_create_account)
        elif (v[len(v)-1]['action'] == 'export_search_mail'):
            v[len(v)-1]['action_length'] = random.choice(export_search_mail)
        elif (v[len(v)-1]['action'] == 'export_bib'):
            v[len(v)-1]['action_length'] = random.choice(export_bib)
        elif (v[len(v)-1]['action'] == 'save_to_multiple_favorites'):
            v[len(v)-1]['action_length'] = random.choice(save_to_multiple_favorites)
        elif (v[len(v)-1]['action'] == 'goto_partner'):
            v[len(v)-1]['action_length'] = random.choice(goto_partner)
        elif (v[len(v)-1]['action'] == 'goto_impressum'):
            v[len(v)-1]['action_length'] = random.choice(goto_impressum)    
        elif (v[len(v)-1]['action'] == 'goto_team'):
            v[len(v)-1]['action_length'] = random.choice(goto_team)
        elif (v[len(v)-1]['action'] == 'goto_contribute'):
            v[len(v)-1]['action_length'] = random.choice(goto_contribute)
        elif (v[len(v)-1]['action'] == 'delete_comment'):
            v[len(v)-1]['action_length'] = random.choice(delete_comment)
        elif (v[len(v)-1]['action'] == 'goto_edit_password'):
            v[len(v)-1]['action_length'] = random.choice(goto_edit_password)
        elif (v[len(v)-1]['action'] == 'goto_delete_account'):
            v[len(v)-1]['action_length'] = random.choice(goto_delete_account)

with open('C:/thesis-goettert/merged_json/merged_file3.json', 'w', encoding='utf-8') as jsonf: 
	jsonf.write(json.dumps(data, indent=4))       