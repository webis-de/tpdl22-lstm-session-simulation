from numpy import genfromtxt
import numpy

my_data= genfromtxt('C:/thesis-goettert/numpyarray/numpyarray.csv', delimiter=',', dtype='int')

new_data = []
 
for x in my_data:
    if (x == my_data[0]).all():
        #print('1',x)
        vorvorgänger = x
    elif (x == my_data[1]).all(): 
        vorgänger = x   
        #print('2',x)
    else:
        #print('3',x)
        if ((vorgänger[0] == x[0]) and (vorvorgänger[0] == x[0])):
            to_append = numpy.concatenate([vorvorgänger,vorgänger,x])
            new_data.append(to_append)
            to_append =[]
            vorvorgänger = vorgänger
            vorgänger = x
        else:
            vorvorgänger = vorgänger
            vorgänger = x

#keep_col = ['nummer1','action_length1','action1','subaction1','origin_action1','request1_1','request1_2','request1_3','response1','nummer2','action_length2','action2','subaction2','origin_action2','request2_1','request2_2','request2_3','response2']
#new_data = new_data[keep_col]

#print(new_data)
#print(len(new_data))
numpy.savetxt('C:/thesis-goettert/numpyarray/numpyarray3.csv', new_data, delimiter=',', fmt='%d')