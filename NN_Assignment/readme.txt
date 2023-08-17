TEAM DETAILS

    1. 22M2119  Kasuba Badri Vishal
    2. 22M108   Aditya Pande
    3. 22M0795  Prathamesh S. Yeole

The data can be downloaded from the following repo : https://github.com/sahasrarjn/cs725-2022-assignment

About Assignment
    1. Implemented all the Functions that were guided from nn.py
    2. Entire Assignment is done self and no malpractises have been done
    3. We declare entire work that is done is through our own understanding of concepts and done from scratch with help of math and numpy libraries
    4. Apart from general requirement, we have implemented couple of optimizers, feature selection, batch normalization and min_max scaling features for training the dataset
    5. Additional task of Classification task is also done with the modification of the original problem as stated in the problem for extra credits

For Execution
    1. nn_1.py, nn_2.py and nn_3.py takes data from ./regression/data directory from the parent directory
    2. nn_classification.py takes data from ./data directory as the python file is already present in ./classification folder where data folder is present


About nn_3.py
    For dimentionality reduction we have used Principal Component Analysis (PCA)
    There fore the new features are the old features projected along the eigenvetors of the covariance matrix obtained from the train data.
    The new features are not the subset of the old one, but rather a transformation, therefore in features.csv the old column names are not used.
    Please note this
    In the features.csv we have written the names of the new features in a general form
