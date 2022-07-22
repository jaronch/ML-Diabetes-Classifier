import numpy as np
import csv 
import math

def pdf(test_data, mean, sd):
    '''
    Helper function that calculates the probability of 
    a numerical attribute given an sd and mean.
    Returns the probability
    '''
    constant = 1/(sd*math.sqrt(2*math.pi))
    exponent = math.exp(-((test_data-mean)**2)/(2*(sd)**2))
    return constant*exponent

def classify_nb(training_filename, testing_filename):
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
    # Open and read files
    training_data = []
    testing_data = []
    with open(training_filename, newline='', encoding='utf-8-sig') as csvfile:
        training_data = list(csv.reader(csvfile))
    
    with open(testing_filename, newline='', encoding='utf-8-sig') as csvfile:
        testing_data = list(csv.reader(csvfile))
    
    final_output_list = []
    yes_data_list = []
    no_data_list = []
    num_yes = 0
    num_no = 0
    num_of_attributes = len(training_data[0]) - 1

    i = 0
    while i < num_of_attributes:
        yes_data_list.append([])
        no_data_list.append([])
        i += 1
    
    
    i = 0
    while i < len(training_data):
        training_list_strings = training_data[i][:-1]
        training_list = [float(s) for s in training_list_strings]
        
        if training_data[i][-1].upper() == "YES":
            num_yes += 1
        else:
            num_no += 1
        
        # divides training examples based off class into yes/no data lists 
        j = 0
        while j < num_of_attributes:
            if training_data[i][-1].lower() == "yes":
                yes_data_list[j].append(training_list[j])
            else:
                no_data_list[j].append(training_list[j])
            j += 1
        i += 1
    
    yes_sd_list = []
    no_sd_list = []
    yes_mean_list = []
    no_mean_list = []

    # Calculate sd and means for attribute given class
    i = 0
    while i < num_of_attributes:
        yes_sd = np.std(yes_data_list[i])
        yes_sd_list.append(yes_sd)
        no_sd = np.std(no_data_list[i])
        no_sd_list.append(no_sd)
        yes_mean = np.mean(yes_data_list[i])
        yes_mean_list.append(yes_mean)
        no_mean = np.mean(no_data_list[i])
        no_mean_list.append(no_mean)
        i += 1

    # Probability of yes/no's needed for Bayes Theorem
    p_yes = num_yes/len(training_data)
    p_no = num_no/len(training_data)
    
    
    i = 0
    while i < len(testing_data):
        test_list_strings = testing_data[i]
        test_list = [float(s) for s in test_list_strings]
        p_yes_cond = 0
        p_no_cond = 0
        p_cond_yes_total = 1
        p_cond_no_total = 1

        # Calculate conditional probability of each attribute
        j = 0
        while j < num_of_attributes:
            p_cond_yes = pdf(test_list[j], yes_mean_list[j], yes_sd_list[j])
            p_cond_no = pdf(test_list[j], no_mean_list[j], no_sd_list[j])
            p_cond_yes_total *= p_cond_yes
            p_cond_no_total *= p_cond_no
            j += 1
        
        p_yes_cond = p_cond_yes_total*p_yes
        p_no_cond = p_cond_no_total*p_no

        # Choose bigger probability as final class for test example
        if p_yes_cond >= p_no_cond:
            final_output_list.append("yes")
        else:
            final_output_list.append("no")
        i += 1

    return final_output_list
