# Learning phase for click stream analysis

import numpy as np
import math

# Convert the test data into numpy array or low medium and high values

def build_test_data(input_test, input_tlabs):
    test_file = open(input_test,'r')
    test_vals = [[int(n) for n in line.split()] for line in test_file]
    test_file.close()
    test_vals = np.array(test_vals)
    
    test_data = []
    tdata = [] 
    
    # No. of features to pick. 
    # we choose the page views of 24 pages to calculate entropy
    
    num_features = 24 #len(train_data[0])
    
    ans_file = open(input_tlabs,'r')
    ans = [int(line) for line in ans_file]
    ans_file.close()
    
    for i in range(num_features):
            tdata.append(test_vals[:,i])
            temp = []
            for item in tdata[i]:
                if 0 < item <= 1:
                    temp.append('low')
                elif 2 <= item <= 4:
                    temp.append('med')
                else:
                    temp.append('high')
            test_data.append(temp)
    test_data.append(ans)
    test_data = np.array(test_data)
    test_data = np.transpose(test_data)
    return test_data
   
# Convert training data to numpy array to be worked on by the second phase

def build_data(input_train, input_labs):
    train_file = open(input_train, 'r')
    train_data = [[int(n) for n in line.split()] for line in train_file]
    train_file.close()

    train_data = np.array(train_data)
    feature_vals = []
    decision_table = []
    num_features = 24 #len(train_data[0])

    class_file = open(input_labs, 'r')
    classifier = [int(line) for line in class_file]
    class_file.close()

    for i in range(num_features):
        feature_vals.append(train_data[:, i])
        temp = []
        for item in feature_vals[i]:
            if 0 < item <= 1:
                temp.append('low')
            elif 2 <= item <= 4:
                temp.append('med')
            else:
                temp.append('high')
        decision_table.append(temp)
    decision_table.append(classifier)

    decision_table = np.array(decision_table)
    decision_table = np.transpose(decision_table)
    return decision_table

# Calculate h(X) value for the training data

def calc_h_to_x(data):
    total_records = len(data[:, -1])
    prob_yes = float(data[:, -1].tolist().count('1')) / float(total_records)
    prob_no = float(data[:, -1].tolist().count('0')) / float(total_records)
    if prob_yes == 0.0:
        return -1  # All no
    if prob_no == 0.0:
        return -2  # All yes
    h_of_x = (-1 * prob_yes * math.log(prob_yes, 2)) - (prob_no * math.log(prob_no, 2))
    return h_of_x

# Calculate information gain using the entropy values and decide which feature to split on

def split_on(h_of_x, decision_table, feature_no_take):
    num_features = len(decision_table[0]) - 1
    ig_of_x_given_y = []
    for i in range(num_features):
    	# Ignore features already splitted on
        if i in feature_no_take:
            ig_of_x_given_y.append(-1)
            continue

        # h(X|high) is calculated here
        num_x_given_no = len(
            decision_table[np.where((decision_table[:, i] == 'high') * (decision_table[:, num_features] == '0'))])
        num_x_given_yes = len(
            decision_table[np.where((decision_table[:, i] == 'high') * (decision_table[:, num_features] == '1'))])
        num_high = len(decision_table[np.where(decision_table[:, i] == 'high')])
        if num_high == 0:
            h_of_x_given_high = 0
        else:
            prob_x_given_yes = float(num_x_given_yes) / float(num_high)
            prob_x_given_no = float(num_x_given_no) / float(num_high)
            if prob_x_given_yes == 0 or prob_x_given_no == 0:
                h_of_x_given_high = 0
            else:
                h_of_x_given_high = (-1 * prob_x_given_yes * math.log(prob_x_given_yes, 2)) - (
                    prob_x_given_no * math.log(prob_x_given_no, 2))

		# h(X|med)
        num_x_given_no = len(
            decision_table[np.where((decision_table[:, i] == 'med') * (decision_table[:, num_features] == '0'))])
        num_x_given_yes = len(
            decision_table[np.where((decision_table[:, i] == 'med') * (decision_table[:, num_features] == '1'))])
        num_med = len(decision_table[np.where(decision_table[:, i] == 'med')])
        if num_med == 0:
            h_of_x_given_med = 0
        else:
            prob_x_given_yes = float(num_x_given_yes) / float(num_med)
            prob_x_given_no = float(num_x_given_no) / float(num_med)
            if prob_x_given_yes == 0 or prob_x_given_no == 0:
                h_of_x_given_med = 0
            else:
                h_of_x_given_med = (-1 * prob_x_given_yes * math.log(prob_x_given_yes, 2)) - (
                    prob_x_given_no * math.log(prob_x_given_no, 2))

		# h(X|low)
        num_x_given_no = len(
            decision_table[np.where((decision_table[:, i] == 'low') * (decision_table[:, num_features] == '0'))])
        num_x_given_yes = len(
            decision_table[np.where((decision_table[:, i] == 'low') * (decision_table[:, num_features] == '1'))])
        num_low = len(decision_table[np.where(decision_table[:, i] == 'low')])
        if num_low == 0:
            h_of_x_given_low = 0
        else:
            prob_x_given_yes = float(num_x_given_yes) / float(num_low)
            prob_x_given_no = float(num_x_given_no) / float(num_low)
            if prob_x_given_yes == 0 or prob_x_given_no == 0:
                h_of_x_given_low = 0
            else:
                h_of_x_given_low = (-1 * prob_x_given_yes * math.log(prob_x_given_yes, 2)) - (
                    prob_x_given_no * math.log(prob_x_given_no, 2))

		# With above values calc IG(X|Y)
        total_records = len(decision_table)
        prob_high = float(num_high) / float(total_records)
        prob_med = float(num_med) / float(total_records)
        prob_low = float(num_low) / float(total_records)
        igval = h_of_x - (
            (prob_high * h_of_x_given_high) + (prob_med * h_of_x_given_med) + (prob_low * h_of_x_given_low))
        ig_of_x_given_y.append(igval)
    # print "ig of x given y : ", ig_of_x_given_y
    
    # Return column with highest information gain
    return ig_of_x_given_y.index(max(ig_of_x_given_y))
