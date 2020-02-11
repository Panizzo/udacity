#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pprint as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

### SKLearn Libraries
# Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

### Classification libraries
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

### Validation libraries
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import PredefinedSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'bonus','deferral_payments', 'deferred_income','exercised_stock_options'
                ,'expenses','long_term_incentive','other','restricted_stock','salary'
                ,'total_payments','total_stock_value','from_messages','from_poi_to_this_person'
                ,'from_this_person_to_poi','shared_receipt_with_poi','to_messages'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Remove TOTAL
data_dict.pop('TOTAL',0)

### Remove features
drop_list = ['director_fees', 'restricted_stock_deferred', 'email_address', 'loan_advances']
### Remove features from dictionary
for person in data_dict:
    for feature in data_dict[person].keys():
        if feature in drop_list:
            del data_dict[person][feature]

### Add features on dictionary
for person in data_dict:

    ### Add perc_exercised_stock
    if (data_dict[person]['exercised_stock_options'] == 'NaN' or data_dict[person]['total_stock_value'] == 'NaN'):
        data_dict[person]['perc_exercised_stock'] = 0
    else:
        data_dict[person]['perc_exercised_stock'] = ( float(data_dict[person]['exercised_stock_options'])
                                                    / float(data_dict[person]['total_stock_value'])) * 100.0

    ### Add perc_bonus
    if (data_dict[person]['bonus'] == 'NaN' or data_dict[person]['total_payments'] == 'NaN'):
        data_dict[person]['perc_bonus'] = 0
    else:
        data_dict[person]['perc_bonus'] = ( float(data_dict[person]['bonus'])
                                          / float(data_dict[person]['total_payments'])) * 100.0

    ### Add perc_message_from_poi
    if (data_dict[person]['from_poi_to_this_person'] == 'NaN' or data_dict[person]['from_messages'] == 'NaN'):
        data_dict[person]['perc_message_from_poi'] = 0
    else:
        data_dict[person]['perc_message_from_poi'] = ( float(data_dict[person]['from_poi_to_this_person'])
                                                     / float( data_dict[person]['from_messages']
                                                            + data_dict[person]['from_poi_to_this_person'])) * 100.0

    ### Add perc_message_to_poi
    if (data_dict[person]['from_this_person_to_poi'] == 'NaN' or data_dict[person]['to_messages'] == 'NaN'):
        data_dict[person]['perc_message_to_poi'] = 0
    else:
        data_dict[person]['perc_message_to_poi'] = ( float(data_dict[person]['from_this_person_to_poi'])
                                                   / float( data_dict[person]['to_messages']
                                                          + data_dict[person]['from_this_person_to_poi'])) * 100.0

### Updating features list
features_list.extend(['perc_exercised_stock'
                     ,'perc_bonus'
                     ,'perc_message_from_poi'
                     ,'perc_message_to_poi'])


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### See Jupyter Notebook for classfiers tests.

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### See Jupyter Notebook for classfiers tests.

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

estimators = [('scale', StandardScaler())
             ,('select', SelectKBest(k=20))
             ,('clf', KNeighborsClassifier(algorithm='ball_tree', leaf_size=10, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=1, p=1,
           weights='uniform'))]

clf = Pipeline(estimators)

features_list = ['poi'
                ,'bonus'
                ,'deferral_payments'
                ,'deferred_income'
                ,'exercised_stock_options'
                ,'expenses'
                ,'long_term_incentive'
                ,'other'
                ,'restricted_stock'
                ,'salary'
                ,'total_payments'
                ,'total_stock_value'
                ,'from_messages'
                ,'from_poi_to_this_person'
                ,'from_this_person_to_poi'
                ,'shared_receipt_with_poi'
                ,'to_messages'
                ,'perc_exercised_stock'
                ,'perc_bonus'
                ,'perc_message_from_poi'
                ,'perc_message_to_poi']

dump_classifier_and_data(clf, my_dataset, features_list)
