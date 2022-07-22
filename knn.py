import csv
import numpy as np

def classify_nn(training_filename, testing_filename, k: int):
    '''
    Input:
    Testing and training data should be inputted in csv format.
    The csv file for testing and training should have the same number of columns (attributes),
    except the training data should have an extra column at the end with a class yes/no.
    The attributes should be presented as numerical values, NOT categorical.

    k should be an integer greater than 0 but less than or equal to the number of examples
    given in the training data

    Output:
    A list of the classes each testing example was classified to with the algorithm
    '''
    # Open and read csv files
    training_data = []
    testing_data = []
    with open(training_filename, newline='', encoding='utf-8-sig') as csvfile:
        training_data = list(csv.reader(csvfile))
    
    with open(testing_filename, newline='', encoding='utf-8-sig') as csvfile:
        testing_data = list(csv.reader(csvfile))
    
    final_output_list = []

    i = 0
    while i < len(testing_data):
        j = 0
        num_yes = 0
        num_no = 0
        distance_list = []

        # Converting test list of strings into numpy array of floats
        test_list_strings = testing_data[i]
        test_list = [float(s) for s in test_list_strings]
        test_array = np.asarray(test_list)
        # Each testing example is compared with all the training examples
        while j < len(training_data):
            distance = 0

            # Converting training list of strings into numpy array of floats
            training_list_strings = training_data[j][:-1]
            training_list = [float(s) for s in training_list_strings]
            training_array = np.asarray(training_list)

            # Numpy method used to calculate euclidian distance
            distance = np.linalg.norm(test_array-training_array)

            # distances and classes for each example stored in list of tuples
            result_tuple = (distance,training_data[j][-1])
            distance_list.append(result_tuple)
            j += 1
        
        distance_list.sort(key=lambda i:i[0])
        
        j = 0
        # Only consider k closest training examples
        while j < k:
            if distance_list[j][1].lower() == "yes".lower():
                num_yes += 1
            else:
                num_no += 1
            j += 1
        
        if num_yes >= num_no:
            final_output_list.append("yes")
        else:
            final_output_list.append("no")

        i += 1         
    return final_output_list