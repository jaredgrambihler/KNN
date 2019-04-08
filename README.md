K-Nearest Neighbors
===================

A simply KNN classifier module built using python, where the K number of neighbors can be specified and run on any dataset.

The data is taken as a tuple of two lists, ([dataList, dataList...], [dataLabel, dataLabel..]), where dataList and dataLabel are ordered (e.g. the first label corresponds to the first element of the data). The data can be a list of any length, while the labels must be single variables.

The data labels should be ints, from 0...n, where n is the number of labels. The classifier will
return the label #, and if the label was, for example, 'elephant', that would have to be converted
to an int for the classifier, and then back to elephant after is has been classified. If the dataset is using non-integers, it may be useful to create a dictionary for conversions.
