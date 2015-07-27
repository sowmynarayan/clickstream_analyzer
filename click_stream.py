#Detection phase for click stream analysis.

import learning
import time
import numpy as np
import copy


class Node:
    def __init__(self):
        self.me = -1
        self.filter_feature = []
        self.filter_values = []
        self.left_child = None  # For 'high'
        self.middle_child = None  # For 'mid'
        self.right_child = None  # For 'low'
        self.answer = -1  # 1 implies leaf

    def restrict_data(self, input_data):
        restricted = input_data
        for feature, value in zip(self.filter_feature, self.filter_values):
            restricted = restricted[np.where(restricted[:, feature] == value)]
        return restricted

    def fill_me(self, input_data):
        h_of_x = learning.calc_h_to_x(input_data)
        self.me = learning.split_on(h_of_x, input_data, self.filter_feature)
        return

    def fill_children(self, input_data):

        # On condition self.filter_feature, self.filter_values split with value self.me
        if self.answer >= 0 or self.me == -1:
            # Nothing further
            return
        if len(self.filter_feature) == len(input_data[0]):
        	ans = self.restrict_data(input_data)[-1]
        	# Answer at leaf
        	self.me = ans
        	return
        left = copy.deepcopy(self)  # DO DEEP COPY !!!
        left.filter_feature.append(self.me)
        left.filter_values.append('low')
        left_data = left.restrict_data(input_data)
        if len(left_data) <= 0:
            # Left empty
            self.left_child = Node()
        else:
            left_h_to_x = learning.calc_h_to_x(left_data)
            if left_h_to_x == -1:  # All no
               
                left.answer = 0
                self.left_child = left
            elif left_h_to_x == -2:  # All yes
               
                left.answer = 1
                self.left_child = left
            else:
                left.me = learning.split_on(left_h_to_x, left_data, left.filter_feature)
                
                self.left_child = left
        # Left done

        middle = copy.deepcopy(self)  # DO DEEP COPY !!!
        middle.filter_feature.append(self.me)
        middle.filter_values.append('med')
        middle_data = middle.restrict_data(input_data)
        if len(middle_data) <= 0:
            # print 'Mid empty'
            self.middle_child = Node()
        else:
            middle_h_to_x = learning.calc_h_to_x(middle_data)
            # print 'middle_h_to_x : ', middle_h_to_x
            if int(middle_h_to_x) == -1:  # All no
                # print 'Mid all no'
                middle.answer = 0
                self.middle_child = middle
            elif int(middle_h_to_x) == -2:  # All yes
                # print 'Mid all yes'
                middle.answer = 1
                self.middle_child = middle
            else:
                middle.me = learning.split_on(middle_h_to_x, middle_data, middle.filter_feature)
                self.middle_child = middle
        #print 'Middle done'

        right = copy.deepcopy(self)  # DO DEEP COPY !!!
        right.filter_feature.append(self.me)
        right.filter_values.append('high')
        right_data = right.restrict_data(input_data)
        if len(right_data) <= 0:
            # print 'Right empty'
            self.right_child = Node()
        else:
            right_h_to_x = learning.calc_h_to_x(right_data)
            if right_h_to_x == -1:  # All yes
                # print 'Right all yes'
                right.answer = 1
                self.right_child = right
            elif right_h_to_x == -2:  # All no
                # print 'Right all no'
                right.answer = 0
                self.right_child = right
            else:
                right.me = learning.split_on(right_h_to_x, right_data, right.filter_feature)
                self.right_child = right
        #print 'Right done'

        # Left child :  self.left_child.me, self.left_child.answer
        # Middle child : self.middle_child.me, self.middle_child.answer
        # Right child :  self.right_child.me, self.right_child.answer
        #print "---------------------------------------------------"
        self.left_child.fill_children(input_data)
        self.middle_child.fill_children(input_data)
        self.right_child.fill_children(input_data)

        return

	# Recursively search the tree and return the leaf value
	
    def search_tree(self, testdata):
        if self.answer >= 0:
            return self.answer
        if self.me == -1:
            return 0
        if testdata[self.me] == 'high':
            return self.left_child.search_tree(testdata)
        elif testdata[self.me] == 'med':
            return self.middle_child.search_tree(testdata)
        elif testdata[self.me] == 'low':
            return self.right_child.search_tree(testdata)
        else:
            print "Wrong data!"
            return -1



if __name__ == '__main__':
    begin_time = time.time()

    root = Node()

    print 'Loading data ...'
    data = learning.build_data('trainfeat.csv', 'trainlabs.csv')
    test_data = learning.build_test_data('testfeat.csv','testlabs.csv')
    print 'Data loaded'
    #print test_data
    root.fill_me(data)
    print "Building the decision tree, please wait..."
    root.fill_children(data)
    print "Decision tree built!"
    
    correct = 0
    total = 0 
    
    f = open('results.txt','w')
    for line in test_data:
        result = root.search_tree(line)
        #print total,"th ans:",result
        f.write(str(result))
        f.write('\n')
        if result == int(line[-1]):
            correct += 1
        total += 1
	
    f.close()
    percent = 100* float(correct)/float(total)
    print percent,"% predictions are correct."  
    print "Please check results.txt for the predictions..."  
    end_time = time.time()
    print "Runtime: ", end_time - begin_time, "s"
