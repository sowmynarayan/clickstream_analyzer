To run the click stream analysis:

1. Ensure the following input files are in same location as the code:
testfeat.csv  testlabs.csv  trainfeat.csv  trainlabs.csv

2. Run the following command:

$ python click_stream.py 

Above will take care of learning from the training data, building the decision tree and then traversing the same.
Runtime is around 1 min.
