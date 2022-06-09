from numpy import genfromtxt
import numpy

my_data= genfromtxt('C:/thesis-goettert/numpyarray/numpyarray2.csv', delimiter=',', dtype='int')

new_data = []
 
for x in my_data:
    if (x == my_data[0]).all():
        vorgänger = x
        new_data.append(x)
    else:
        if vorgänger[0] != x[0]:
            new_data.append(x)
            vorgänger = x



numpy.savetxt('C:/thesis-goettert/numpyarray/first_step.csv', new_data, delimiter=',', fmt='%d')