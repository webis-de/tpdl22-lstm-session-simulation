import csv 
import json 
import os
from glob import glob

# get all csv file path from my directory and put them in a list
PATH = "C:/thesis-goettert/data_/"
EXT = "*.csv"
all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]	

# iterate over every single csv file
counter = 0
for i in all_csv_files:
	# progress in %
	print(counter/262428)
	counter += 1
	# create a dictionary and other helper dictionaries 
	dictionary = {}
	log_id_dict = {}
	request = {}
	response = {}
	old_log_id = 0
	old_key = ""
	previous_action = ''
	continue_loop = True

	# create key and add it to the dict
	def set_key(key):
		global old_key
		if(old_key != key):
			dictionary.update({key: []})
			old_key = key

	# fill the dict by extracting all valuable information from a given row
	def extract_information(log_id, row):
		global old_log_id, dictionary, log_id_dict, request, response, previous_action, continue_loop
		
		# extract the additional information if the log_id has already been used
		if (old_log_id == log_id):

			# fill action_substate with more valuable information to specify the previous action
			if(row['mapping_type'] == 'action'):	
				log_id_dict['action_substate'].append(row['mapping_action_label'])

			# add every extraction that is not resultlistids or single doc(docid) to the request dict, put the resultlistids as list in the response dict
			if (row['mapping_type'] == 'extraction'):
				
				if(not((row['mapping_action_label'] == 'resultlistids') | (row['mapping_action_label'] == 'docid'))):
					if '?daterange[]' in (row['params']):
						x = row['params'].split('?daterange[]')
						request.update({row['mapping_action_label']: x[0]})
						request.update({'daterange': x[1]})
					elif '?filter[]' in (row['params']):
						y = row['params'].split('?filter[]')
						request.update({row['mapping_action_label']: y[0]})
						request.update({'filter': y[1]})
					else:		
						request.update({row['mapping_action_label']: row['params']})

				else:
					# clean the Strings
					if(row['mapping_action_label'] == 'resultlistids'):
						if(len(row['params']) == 255):
							a = ','.join(row['params'].split(',')[:-1])
						else:
							a = ','.join(row['params'].split(','))	
						a = a.split(',')
						response.update({row['mapping_action_label']: a})
					else: 
						response.update({row['mapping_action_label']: row['params']})	
		# extract information from a log_id that has his first appearance in the key			
		else:
			# special case if a log_id starts with extraction that is a resultlistids
			if(((row['mapping_type'] == 'extraction' ) & (row['mapping_action_label'] == 'resultlistids')) | ((row['mapping_type'] == 'extraction' ) & (row['mapping_action_label'] == 'sort_param')) | ((row['mapping_type'] == 'extraction' ) & (row['mapping_action_label'] == 'searchterm_2')) | ((row['mapping_type'] == 'extraction' ) & (row['mapping_action_label'] == 'informationtype_param')) | ((row['mapping_type'] == 'extraction' ) & (row['mapping_action_label'] == 'page_param'))):
				continue_loop = False
				return
			# update old_log_id
			old_log_id = log_id

			# if there is data stored from a previous log_id append request and response to log_id_dict and the append log_id_dict to the dictionary key
			if(log_id_dict != {}):
				#delete searchterm2 if searchterm1 is already in request
				if (('searchterm_1' in request) and ('searchterm_2' in request)):
					del request['searchterm_2']
				if (('searchterm_2' in request) and ('searchterm_3' in request)):
					del request['searchterm_3']	
				log_id_dict.update({"request": request})
				log_id_dict.update({"response": response})
				#dont add session when request contains only informationtype_param or page_param
				if(((len(request) > 0) and (('searchterm_1' in request) or ('searchterm_2' in request) or ('searchterm_3' in request) or ('searchterm_4' in request))) or (len(request) == 0)):
					dictionary[row['key']].append(log_id_dict)

			# after appending you hae to reset all helper dictionaries	
			log_id_dict = {}
			request = {}
			response = {}

			# fill the log_id_dict with all the data that doesnt change for a log_id
			log_id_dict.update({"log_id": row['log_id'],"part_length": row['part_length'],"date": row['date'],"action_length": row['action_length'],"part_step": row['part_step']})
			
			# add action data to the log_id_dict
			if (row['mapping_type'] == 'action'):
				log_id_dict.update({"action": row['mapping_action_label']})	

			# check for special case
			if 'action' not in log_id_dict:
				if ((row['mapping_type'] == 'extraction') & (row['mapping_action_label'] == 'docid')):
					log_id_dict.update({"action": 'docid'})	

			#special case were the action view_record is shown as view_record_rec
			if(log_id_dict['action'] == 'view_record_rec'):
				log_id_dict.update({"action": 'view_record'})	

			# add action_substate list to the log_id_dict it gets filled in the extract_information function 
			# it gets filled if there are multiple actions in one log_id
			log_id_dict.update({'action_substate': []})

			# origin_action takes the action from last log_id
			# this describes the last action from the previous log_id
			log_id_dict.update({"origin_action": previous_action})

			# store action from this session for the origin action of the next log_id
			previous_action = log_id_dict['action']
			
			# add every extraction that is not resultlistids or single doc(docid) to the request dict, put the resultlistids as list in the response dict
			if (row['mapping_type'] == 'extraction'):
				
				if(not((row['mapping_action_label'] == 'resultlistids') | (row['mapping_action_label'] == 'docid'))):
					if '?daterange[]'.isin(row['params']):
						x = row['params'].split('?daterange[]')
						request.update({row['mapping_action_label']: x[0]})
						request.update({'daterange': x[1]})
					elif '?filter[]'.isin(row['params']):
						y = row['params'].split('?filter[]')
						request.update({row['mapping_action_label']: y[0]})
						request.update({'filter': y[1]})
					else:		
						request.update({row['mapping_action_label']: row['params']})

				else:
					# clean the Strings
					if(row['mapping_action_label'] == 'resultlistids'):
						if(len(row['params']) == 255):
							a = ','.join(row['params'].split(',')[:-1])
						else:
							a = ','.join(row['params'].split(','))	
						a = a.split(',')
						response.update({row['mapping_action_label']: a})
					else: 
						response.update({row['mapping_action_label']: row['params']})				

	# Function to convert a CSV to JSON 
	# Takes the file paths as arguments 
	def make_json(csvFilePath, jsonFilePath):  
		global continue_loop
		# Open a csv reader called DictReader 
		with open(csvFilePath, encoding='utf-8') as csvf: 
			csvReader = csv.DictReader(csvf) 
			# iterate over every single row
			
			for row in csvReader: 
					if(continue_loop):
						# give key to function set_key(key)
						key = row['key']
						set_key(key)

						# give log_id and complete row data to extract_information
						log_id = row['log_id'] 
						extract_information(log_id, row)
		if(continue_loop):
			#delete searchterm2 if searchterm1 is already in request
			if (('searchterm_1' in request) and ('searchterm_2' in request)):
				del request['searchterm_2']
			if (('searchterm_2' in request) and ('searchterm_3' in request)):
					del request['searchterm_3']		
			# append last log_id to the key 
			if(((len(request) > 0) and (('searchterm_1' in request) or ('searchterm_2' in request) or ('searchterm_3' in request) or ('searchterm_4' in request))) or (len(request) == 0)):
					dictionary[row['key']].append(log_id_dict)

			log_id_dict.update({"request": request})
			log_id_dict.update({"response": response})
			#dictionary[row['key']].append(log_id_dict)

			# Open a json writer, and use the json.dumps() 
			# function to dump dictionary 
			with open(jsonFilePath, 'w', encoding='utf-8') as jsonf: 
				jsonf.write(json.dumps(dictionary,ensure_ascii=False, indent=4)) 

		continue_loop = True		

	# Driver Code 			 
	# Decide the two file paths according to your 
	# computer system
	csvFilePath = i

	# create json file path from csv file path
	j = i.replace('/data_', '/json')
	j = j.replace('.csv', '.json')
	jsonFilePath = j

	# Call the make_json function 
	make_json(csvFilePath, jsonFilePath)