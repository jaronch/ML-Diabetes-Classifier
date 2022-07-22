# Machine Learning Diabetes Classifier
## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Further info](#further-info)
* [Setup](#setup)
* [Project Status](#project-status)
* [Sources](#sources)


## General info
k Nearest Neighbour and Naive Bayes Machine Learning Algorithms to classify whether testing data has diabetes, using the Pima Indians Diabetes Database.
It was sourced from the National Institute of Diabetes and Digestive and Kidney Diseases first released 9 May, 1990. It’s first usage was in Smith et al.’s paper ‘Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus’ (1988). 

knn.py and naive.py house the respective functions that can be used for any nominal dataset in csv format with training and testing data. final_test.py is a program that has these algorithms classify on the Pima Indian Diabetes Dataset that has gone through 10-fold cross-validation, and prints the accuracy of classification of the algorithms to the screen.

## Technologies
Project is created with:
* python3
* Weka - Machine Learning Software used to split files and also cross-check correctness of my algorithms completed from scratch.
	
## Setup
To run this project which tests the accuracy of the algorithms on this dataset, have python3 installed and run:

```
$ python3 final_test.py
```
## Further info
The data has been slightly modified by replacing missing values with averages, and making all the input values nominal. In addition, the final_test.py program runs on csv and txt files of the same dataset that have gone through 10-fold cross-validation. For each fold, there is a separate 'training%d.csv', 'testing%d.csv' and 'testing_r%d.txt' file where %d are the integers from 1-10. This process of creating 10-fold cross-validated files was achieved with Weka Software. This project will soon develop the program further to do the process itself. 

The final_test.py program will print to standard output the accuracy results of our classifiers. On this dataset with 10-fold cross-validation, my Naive Bayes Algorithm achieves an accuracy of 75.00%, and for the kNN algorithm, when k = 1 an accuracy of 69.27% is achieved, and for k = 5 an accuracy of 75.52% is achieved.

My k Nearest Neighbour (kNN) Algorithm is housed in knn.py as the function classify_nn(training_filename, testing_filename, k: int). It is a supervised Machine Learning Algorithm that calculates the euclidian distance of each training example with a testing example, then classifies that testing example by ordering the distance in ascending order. The classes of the k closest training examples are considered, with the testing example being classified by the majority class.

My Naive Bayes (NB) Algorithm is housed in naive.py as the function classify_nb(training_filename, testing_filename). It is a supervised Machine Learning Algorithm that calculates the conditional probability given the class yes/no of the testing examples in relation to the sd and mean of the training examples. Each conditional probability per-attribute is calculated for class yes and no and multipled together. The bigger conditional probability leads to the result of the classification as yes or no. 

## Project Status
Algorithms functional! Accuracy test functional! Will further develop program to be able to split data into 10 fold cross-validation, rather than using Weka to create the cross-validated files.

## Sources
Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C. and Johannes, R.S. (1988) Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus. Proceedings of the Annual Symposium on Computer Application in Medical Care, Orlando, 7-11 November, 261-265.

Eibe Frank, Mark A. Hall, and Ian H. Witten (2016). The WEKA Workbench. Online Appendix for "Data Mining: Practical Machine Learning Tools and Techniques", Morgan Kaufmann, Fourth Edition, 2016.
 
