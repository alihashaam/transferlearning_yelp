Notes on using SVMs for large scale problems (Dataset: YELP challenge 10):
==========================================================================

1- The first tip presents the justification for using SGD instead of the regular SVC or LinearSVC for large scale data:
http://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use

2- This is the page of SGDClassifier (member of SVM family)
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier

3- I tried to use both scaled and unscaled data for SGDClassifer (because SVMs are sensitive to variant data, as stated in point 1). Contrary to expectations, the unscaled data gave better results than the scaled data by an improvement of 1%. I'll stick with the scaled version to be consitnet with the requirments of SVM.
