import numpy as np
import csv 
import math
from knn import *
from naive import *


def accuracy_test(func, k = 1):
    avg_sum = 0
    i = 1
    while i <= 10:
        testing_r_data = []
        testing_r_string = './testing_files/testing_r' + str(i) + '.txt'
        with open(testing_r_string) as f:
            testing_r_data = f.read().splitlines()

        training_file_string = './training_files/training' + str(i) + '.csv'
        testing_file_string = './testing_files/testing' + str(i) + '.csv'
        output_list = []
        if func == classify_nb:
            output_list = func(training_file_string, testing_file_string)
        else:
            output_list = func(training_file_string, testing_file_string, k)
        
        j = 0
        correct_sum = 0
        while j < len(testing_r_data):
            if testing_r_data[j] == output_list[j]:
                correct_sum += 1
            j += 1
        final_avg = correct_sum/len(testing_r_data)
        avg_sum += final_avg 
        print('Fold %d: %f' %(i, final_avg))
        i += 1
    final_final_avg = avg_sum/10
    print('Final average: %f' %final_final_avg)


print('All decimals provided show percentage values on the accuracy of classification by the algorithm')
print('10 fold cross validation with no CFS')
print()
print('Naive Based Solution:')
accuracy_test(classify_nb)
print()
print('1NN Solution:')
accuracy_test(classify_nn, 1)
print()
print('5NN Solution:')
accuracy_test(classify_nn, 5)
