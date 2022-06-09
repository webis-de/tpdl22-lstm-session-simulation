import json
import numpy
import sys
from collections import Counter
import statistics
import matplotlib.pyplot as plt
import numpy as np
#import itertools

f = open('C:/thesis-goettert/merged_json/merged_file3.json')
data = json.load(f)

überliste = []
liste = []
nummer = 0
actionsubstate = []
actionlength = []
searchtermlength = []
pageparamlength = []
responselength = []
actionlengths = []
responserich = []
responserich2 = []
page = []
sorted_actions = []


for (k,v) in data.items():
    actionlengths.append((len(v)))
    for i in v:
        for(k2,v2) in i.items():
            if(k2 == 'action_length'):
                if(int(v2) <= 200):
                    liste.append(int(v2))
                    actionlength.append(int(v2))
                else:
                    liste.append(200)    
                    actionlength.append(200)
            if(k2 == 'action'):
                action = v2
                if(v2 == 'export_bib'):
                    liste.append(0)
                elif(v2 == 'export_cite'):
                    liste.append(1) 
                elif(v2 == 'export_search_mail'):
                    liste.append(2)
                elif(v2 in ('goto_about','goto_contribute','goto_impressum','goto_partner','goto_sofis','goto_team')):
                    liste.append(3)
                elif(v2 == 'goto_advanced_search'):
                    liste.append(4)
                elif(v2 in('goto_create_account', 'goto_delete_account', 'goto_edit_password')):
                    liste.append(5)
                elif(v2 == 'goto_favorites'):
                    liste.append(6)
                elif(v2 == 'goto_fulltext'):
                    liste.append(7)
                elif(v2 == 'goto_google_books'):
                    liste.append(8)
                elif(v2 == 'goto_google_scholar'):
                    liste.append(9)
                elif(v2 == 'goto_history'):
                    liste.append(10)
                elif(v2 == 'goto_home'):
                    liste.append(11)
                elif(v2 == 'goto_local_availability'):
                    liste.append(12)
                elif(v2 == 'goto_login'):
                    liste.append(13)
                elif(v2 == 'goto_topic-feeds'):
                    liste.append(14)
                elif(v2 == 'goto_topic-research'):
                    liste.append(15)
                elif(v2 == 'save_search'):
                    liste.append(16)
                elif(v2 == 'save_to_multiple_favorites'):
                    liste.append(17)
                elif(v2 == 'search'):
                    liste.append(18)
                elif(v2 == 'search_advanced'):
                    liste.append(19)
                elif(v2 == 'search_institution'):
                    liste.append(20)
                elif(v2 == 'search_keyword'):
                    liste.append(21)
                elif(v2 == 'search_person'):
                    liste.append(22)
                elif(v2 == 'view_record'):
                    liste.append(23)
                elif(v2 == 'query_form'):
                    liste.append(24)
                elif(v2 == 'docid'):
                    liste.append(25)
                elif(v2 == 'goto_thesaurus'):
                    liste.append(26)
                elif(v2 == 'delete_comment'):
                    liste.append(27)
                elif(v2 == ''):
                    liste.append(28)       
            if(k2 == 'action_substate'):
                if(v2 == []):
                    liste.append(0)
                elif(v2[0] == 'view_comment'):
                    liste.append(2)
                elif(v2[0] == 'view_description'):
                    liste.append(3)
                elif(v2[0] == 'view_citation'):
                    liste.append(4)
                elif(v2[0] == 'to_favorites'):
                    liste.append(5)
                elif(v2[0] == 'search_change_only_fulltext'):
                    liste.append(6)
                elif(v2[0] == 'search'):
                    liste.append(7)
                elif(v2[0] == 'search_thesaurus'):
                    liste.append(8)
                elif(v2[0] == 'search_change_sorting'):
                    liste.append(9)
                elif(v2[0] == 'goto_advanced_search_reconf'):
                    liste.append(10)
                elif(v2[0] == 'purge_history'):
                    liste.append(11)
                elif(v2[0] == 'search_from_history'):
                    liste.append(12)
                elif(v2[0] == 'save_search_history'):
                    liste.append(13)
                elif(v2[0] == 'export_mail'):
                    liste.append(14)
                elif(v2[0] == 'search_change_only_fulltext_2'):
                    liste.append(15)
                elif(v2[0] == 'goto_last_search'):
                    liste.append(16)
                elif(v2[0] == 'view_references'):
                    liste.append(17)
                elif(v2[0] == 'export_bib'):
                    liste.append(18)
                elif(v2[0] == 'search_change_nohts'):
                    liste.append(19)
                elif(v2[0] == 'search_change_nohts_2'):
                    liste.append(20)
                elif(v2[0] == 'goto_topic-research-unique'):
                    liste.append(21)
                elif(v2[0] == 'goto_team'):
                    liste.append(22)
                elif(v2[0] == 'goto_about'):
                    liste.append(23)
                elif(v2[0] == 'search_as_rss'):
                    liste.append(24)
                elif(v2[0] == 'goto_fulltext'):
                    liste.append(25)
                elif(v2[0] == 'goto_partner'):
                    liste.append(26)
                elif(v2[0] == 'goto_login'):
                    liste.append(27)
                elif(v2[0] == 'search_person'):
                    liste.append(28)
                elif(v2[0] == 'query_form'):
                    liste.append(29)     
                elif(v2[0] == 'search_change_paging'):
                    liste.append(30)
                elif(v2[0] == 'search_change_facets'):
                    liste.append(31)  
                elif(v2[0] == 'view_record'):
                    liste.append(32) 
                elif(v2[0] == 'goto_google_books'):
                    liste.append(33)
                elif(v2[0] == 'goto_google_scholar'):
                    liste.append(34)
                elif(v2[0] == 'export_cite'):
                    liste.append(35)
                elif(v2[0] == 'goto_home'):
                    liste.append(36)
                elif(v2[0] == 'goto_local_availability'):
                    liste.append(37)
                elif(v2[0] == 'search_institution'):
                    liste.append(38)
                elif(v2[0] == 'goto_favorites'):
                    liste.append(39)
                elif(v2[0] == 'docid'):
                    liste.append(40)
                elif(v2[0] == 'goto_advanced_search'):
                    liste.append(41)
                elif(v2[0] == 'goto_thesaurus'):
                    liste.append(42)
                elif(v2[0] == 'goto_sofis'):
                    liste.append(43)
                elif(v2[0] == 'goto_history'):
                    liste.append(44)
                elif(v2[0] == 'search_keyword'):
                    liste.append(45)
                elif(v2[0] == 'goto_topic-feeds'):
                    liste.append(46)
                elif(v2[0] == 'goto_topic-research'):
                    liste.append(47)
                elif(v2[0] == 'goto_create_account'):
                    liste.append(48)
                elif(v2[0] == 'save_to_multiple_favorites'):
                    liste.append(49)
                elif(v2[0] == 'export_search_mail'):
                    liste.append(50)
                elif(v2[0] == 'goto_impressum'):
                    liste.append(51)
                elif(v2[0] == 'goto_contribute'):
                    liste.append(52)
                elif(v2[0] == 'save_search'):
                    liste.append(53)
                elif(v2[0] == 'delete_comment'):
                    liste.append(54)
                elif(v2[0] == 'goto_edit_password'):
                    liste.append(55)
                elif(v2[0] == 'goto_delete_account'):
                    liste.append(56)
                elif(v2[0] == 'search_advanced'):
                    liste.append(57)
                                                                                                                                                                                                                                                                      
            if(k2 == 'origin_action'):
                if(v2 == 'export_bib'):
                    liste.append(0)
                elif(v2 == 'export_cite'):
                    liste.append(1) 
                elif(v2 == 'export_search_mail'):
                    liste.append(2)
                elif(v2 in ('goto_about','goto_contribute','goto_impressum','goto_partner','goto_sofis','goto_team')):
                    liste.append(3)
                elif(v2 == 'goto_advanced_search'):
                    liste.append(4)
                elif(v2 in('goto_create_account', 'goto_delete_account', 'goto_edit_password')):
                    liste.append(5)
                elif(v2 == 'goto_favorites'):
                    liste.append(6)
                elif(v2 == 'goto_fulltext'):
                    liste.append(7)
                elif(v2 == 'goto_google_books'):
                    liste.append(8)
                elif(v2 == 'goto_google_scholar'):
                    liste.append(9)
                elif(v2 == 'goto_history'):
                    liste.append(10)
                elif(v2 == 'goto_home'):
                    liste.append(11)
                elif(v2 == 'goto_local_availability'):
                    liste.append(12)
                elif(v2 == 'goto_login'):
                    liste.append(13)
                elif(v2 == 'goto_topic-feeds'):
                    liste.append(14)
                elif(v2 == 'goto_topic-research'):
                    liste.append(15)
                elif(v2 == 'save_search'):
                    liste.append(16)
                elif(v2 == 'save_to_multiple_favorites'):
                    liste.append(17)
                elif(v2 == 'search'):
                    liste.append(18)
                elif(v2 == 'search_advanced'):
                    liste.append(19)
                elif(v2 == 'search_institution'):
                    liste.append(20)
                elif(v2 == 'search_keyword'):
                    liste.append(21)
                elif(v2 == 'search_person'):
                    liste.append(22)
                elif(v2 == 'view_record'):
                    liste.append(23)
                elif(v2 == 'query_form'):
                    liste.append(24)
                elif(v2 == 'docid'):
                    liste.append(25) 
                elif(v2 == 'goto_thesaurus'):
                    liste.append(26)
                elif(v2 == 'delete_comment'):
                    liste.append(27)    
                elif(v2 == ''):
                    liste.append(28)              

            if(k2 == 'request'):
                if(v2 == {}):
                    liste.append(0)
                    liste.append(0)
                    liste.append(0)
                    liste.append(0)
                    liste.append(0)
                    liste.append(0)
                else:
                    sortparam = False
                    pageparam = False
                    informationtypeparam = False
                    for k3,v3 in v2.items():   
                            if k3 == 'searchterm_1':
                                liste.append(1)
                                if(len(v3) <= 100):
                                    liste.append(len(v3))
                                    searchtermlength.append(len(v3))
                                else:
                                    liste.append(100)
                                    searchtermlength.append(100)
                                if(('OR' in v3) or ('AND' in v3)): 
                                    liste.append(1)
                                else:
                                    liste.append(0)
                            elif k3 == 'searchterm_2':
                                liste.append(2)
                                if(len(v3) <= 100):
                                    liste.append(len(v3))
                                    searchtermlength.append(len(v3))
                                else:
                                    liste.append(100)
                                    searchtermlength.append(100)
                                if(('OR' in v3) or ('AND' in v3)): 
                                    liste.append(1)
                                else:
                                    liste.append(0)   
                            elif k3 == 'searchterm_3':
                                liste.append(3)
                                if(len(v3) <= 100):
                                    liste.append(len(v3))
                                    searchtermlength.append(len(v3))
                                else:
                                    liste.append(100)    
                                    searchtermlength.append(100)
                                if(('OR' in v3) or ('AND' in v3)): 
                                    liste.append(1)
                                else:
                                    liste.append(0)     
                            elif k3 == 'searchterm_4':
                                liste.append(4)
                                if(len(v3) <= 100):
                                    liste.append(len(v3))
                                    searchtermlength.append(len(v3))
                                else:
                                    liste.append(100)
                                    searchtermlength.append(100)
                                if(('OR' in v3) or ('AND' in v3)): 
                                    liste.append(1)
                                else:
                                    liste.append(0) 
                            elif(k3 == 'sort_param'):
                                sortparam = True    
                            elif(k3 == 'page_param'):
                                if('&view=list' in v3):
                                      pageparam_number = v3.replace('&view=list', '')
                                else:
                                    pageparam_number = v3  
                                pageparam = True    
                            elif(k3 == 'informationtype_param'):
                                informationtypeparam = True
                                if('project' in v3):
                                    informationtypeparam_string = 1
                                elif('literature' in v3):
                                    informationtypeparam_string = 2    
                                    

                    if(sortparam):
                        liste.append(1)
                        sorted_actions.append(action)
                    else: liste.append(0)
                    if(pageparam):
                        page.append(action)
                        if(int(pageparam_number) <= 20):
                            liste.append(int(pageparam_number))
                            pageparamlength.append(int(pageparam_number))
                        else: 
                            liste.append(20)    
                            pageparamlength.append(20)
                    else: liste.append(0)
                    if(informationtypeparam):
                        liste.append(informationtypeparam_string)
                    else: liste.append(0)                


            if(k2 == 'response'):
                if (v2 == {}):
                    liste.append(0)
                for z in v2:
                    for k4,v4 in v2.items():
                        if k4 == 'resultlistids':
                            liste.append(len(v4))
                            responselength.append(len(v4))
                            responserich.append(action)
                        elif k4 == 'docid':
                            liste.append(1)   
                            responselength.append(1)
                            responserich2.append(action)            

        überliste.append(liste)
        liste = []
    nummer += 1    

print(Counter(sorted(actionlengths)))
actionlengths = Counter(sorted(actionlengths))
actionlength_plot, numbers = zip(*actionlengths.items())
#indexes = np.arange(len(labels))

axes = plt.gca()
#axes.set_ylim([0,1000])
axes.set_xlim([0,25])

#plt.xlabel('actionlength')
#plt.ylabel('number of actionlegths')
#plt.plot(actionlength_plot, numbers, 'ro')
#plt.show()
#plt.savefig('C:/thesis-goettert/data/actionlength_plot.png')

responserich = list(dict.fromkeys(responserich))
responserich2 = list(dict.fromkeys(responserich2))
page = list(dict.fromkeys(page))
sorted_actions = list(dict.fromkeys(sorted_actions))

print(responserich)
print(responserich2)
print(page)
print(sorted_actions)



#print(min(actionlength),max(actionlength),statistics.mean(actionlength),'actionlength')
#print(min(searchtermlength),max(searchtermlength),statistics.mean(searchtermlength),'searchtermlength')
#print(min(pageparamlength),max(pageparamlength),statistics.mean(pageparamlength),'pageparamlength')
#print(min(responselength),max(responselength),statistics.mean(responselength),'responselength')

#actionlength = sorted(actionlength)
#searchtermlength = sorted(searchtermlength)
#pageparamlength = sorted(pageparamlength)
#responselength = sorted(responselength)

#with open("C:/thesis-goettert/numpyarray/actionlength.txt", "w") as output:
 ##   output.write(str(actionlength))

#with open("C:/thesis-goettert/numpyarray/searchtermlength.txt", "w") as output:
   # output.write(str(searchtermlength))

#with open("C:/thesis-goettert/numpyarray/pageparamlength.txt", "w") as output:
   # output.write(str(pageparamlength))

#with open("C:/thesis-goettert/numpyarray/responselength.txt", "w") as output:
    #output.write(str(responselength))            

#newarray = numpy.array(überliste, dtype=int)
#a = numpy.asarray(newarray)
#numpy.savetxt('C:/thesis-goettert/numpyarray/numpyarray.csv', a, delimiter=',', fmt='%d')

#print(a)
#print(len(a))
#i = 0
#for x in überliste:
 #   if(len(x) < 9):
 #       print(x)
 #       if(i == 1):
 #           sys.exit()    
#        i += 1   
#print(überliste) 