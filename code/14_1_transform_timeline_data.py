import numpy as np
import pandas as pd
import sys

timeline_data = pd.read_csv('C:/thesis-goettert/numpyarray/session_length_15.csv', names=['session','action','action_length','request1_4','request1_5','response'])

k = 0
z = 1
while k < 186:
    session = timeline_data.loc[timeline_data['session'] == k]

    event = session['action']
    request1_5 = session['request1_5']
    event2 = session['response']
    sort_param = session['request1_4']
    for i,row in event.iteritems():
        if(row == 0):
            event[i] = 'export_bib'
        elif(row == 1):
            event[i] = 'export_cite'
        elif(row == 2):
            event[i] = 'export_search_mail'
        elif(row == 3):
            event[i] = 'goto_about','goto_contribute','goto_impressum','goto_partner','goto_sofis','goto_team'
        elif(row == 4):
            event[i] = 'goto_advanced_search'
        elif(row == 5):
            event[i] = 'goto_create_account', 'goto_delete_account', 'goto_edit_password'
        elif(row == 6):
            event[i] = 'goto_favorites'
        elif(row == 7):
            event[i] = 'goto_fulltext'
        elif(row == 8):
            event[i] = 'goto_google_books'
        elif(row == 9):
            event[i] = 'goto_google_scholar'
        elif(row == 10):
            event[i] = 'goto_history'
        elif(row == 11):
            event[i] = 'goto_home'
        elif(row == 12):
            event[i] = 'goto_local_availability'
        elif(row == 13):
            event[i] = 'goto_login'
        elif(row == 14):
            event[i] = 'goto_topic-feeds'
        elif(row == 15):
            event[i] = 'goto_topic-research'
        elif(row == 16):
            event[i] = 'save_search'
        elif(row == 17):
            event[i] = 'save_to_multiple_favorites'
        elif(row == 18):
            event[i] = 'search'
            if(sort_param[i] == 0):
                sort_param[i] = 'unsorted'
            else: sort_param[i] = 'sorted'
        elif(row == 19):
            event[i] = 'search_advanced'
            if(sort_param[i] == 0):
                sort_param[i] = 'unsorted'
            else: sort_param[i] = 'sorted'
        elif(row == 20):
            event[i] = 'search_institution'
        elif(row == 21):
            event[i] = 'search_keyword'
        elif(row == 22):
            event[i] = 'search_person'
        elif(row == 23):
            event[i] = 'view_record'
            event2[i] = 1
        elif(row == 24):
            event[i] = 'query_form'                                                                                                
        elif(row == 25):
            event[i] = 'docid'
            event2[i] = 1
        elif(row == 26):
            event[i] = 'goto_thesaurus'
        elif(row == 27):
            event[i] = 'delete_comment'
        elif(row == 27):
            event[i] = 'empty'            
        #print(i, row)   

    length2 = session['action_length']
    begin = [0]
    end = []
    length = []
    length_list = length2.tolist()

    for i in range(15):
        if i != 0:
            begin.append(begin[i-1] + length_list[i-1])

    for i in range(15):
        if i == 0:
            end.append(0 + length_list[i])
        else:
            end.append(end[i-1] + length_list[i])  

    for i in range(15):
        length.append(length_list[i])              


    begin = pd.DataFrame(np.array(begin),columns= ['a'])
    end = pd.DataFrame(np.array(end),columns= ['a'])
    length = pd.DataFrame(np.array(length),columns= ['a'])
    begin = begin['a']
    end = end['a']
    length = length['a']

    x = [0,7,27,57,73,89,105,121,129,137,145,153,161,169,177,185]
    if k in x:
        z += 1
    begin = begin.head(z)
    end = end.head(z)
    length = length.head(z)


    levels = np.tile([4,-4,3,-3,2,-2,1,-1],
                    int(np.ceil(len(begin)/4)))[:len(begin)]

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    plt.figure(figsize=(14,8))

    plt.barh(0, (end-begin), color='aqua', height =0.3 ,left=begin, edgecolor = "black")
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    plt.title('Session', fontsize = '25')
    plt.xlabel('Time in seconds', fontsize = '20')
    # plt.yticks(range(len(begin)), "")
    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)
    plt.xlim(-20, end[len(end)-1] + 20)
    plt.ylim(-5,5)
    plt.vlines(begin+length/2, 0, levels, color="tab:red")
    for i in range(len(end)):
        #t = ('event.iloc[i],\n'
        #    'int(event2.iloc[i])\n'
         #   'int(request1_5.iloc[i])\n')
        if(event.iloc[i] in ['search_advanced', 'search']):
            plt.text(begin.iloc[i] + length.iloc[i]/2, 
                levels[i]*1.1,(str(event.iloc[i]) + '/resp:' + str(int(event2.iloc[i])) + '/page:' + str(int(request1_5.iloc[i])) + '/' + sort_param.iloc[i]), 
                ha='center', fontsize = '12')
        elif(event.iloc[i] in ['goto_advanced_search', 'goto_history', 'goto_home', 'search_institution', 'search_keyword', 'search_person', 'view_record', 'docid']):
            plt.text(begin.iloc[i] + length.iloc[i]/2, 
                levels[i]*1.1, (str(event.iloc[i]) + '/resp:' + str(int(event2.iloc[i]))), 
                ha='center', fontsize = '12')
        else:
             plt.text(begin.iloc[i] + length.iloc[i]/2, 
                levels[i]*1.1, (str(event.iloc[i])), 
                ha='center', fontsize = '12')       

    plt.tight_layout()
    #plt.show()
    plt.savefig('C:/thesis-goettert/sessions/session' + str(k) + '.png')
    k += 1