Classifier can be set to work with any set of input data.
The data is taken as a tuple list, ([dataList, dataList...], [dataLabel, dataLabel..])
Where dataList and dataLabel order are the same.
K number of neighbors can be specified as well.

The data labels must be ints, from 0...n, where n is the number of labels. The classifier will
return the label #, and if the label was, for example, 'elephant', that would have to be converted
to an int for the classifier, and then back to elephant after is has been classified
