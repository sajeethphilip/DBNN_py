The Difference Boosting Neural Network (DBNN) published in  Intelligent Data Analysis, 4(2000) 463-473, IOS Press, nl is a simple Bayesian network that applies an imposed conditional independence of joint probability of multiple features
for the classification of the feature space overcoming the limitations of the Naive Bayesian networks while at the same time maintaining its simplicity. The original code was in C++ and was used only by a selected group of people. 
The Python version is a rewrite in just a few lines of code with examples from the UCI repository that can be downloaded and used to create training/test data to evaluate the model quickly. A manual is in preparation.

You can generate config files for UCI data (that can be used as an example) and get the required data for evaluation using the Generate and Get functions. NOTE: You do not need to download the data unless you want to look at it in detail. The config data has the path to the file and can be directly used by the code to get the data.

Add a # to the configure file features before the feature names to exclude any feature from the computation. By default, any repetitions with high cardinality will be filtered out.

By default, the Imposed conditional independence is assumed on pairs of features ("feature_group_size": 2) but can be edited in the conf file to any number equal/less than the total number of features.

Press "q" or "Q" key during training to interrupt training and go to the next stage.

The older version of the C++ code used space as the separator, which is not ideal. space2csv.py function can convert the file into CSV.
