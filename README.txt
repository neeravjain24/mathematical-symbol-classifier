##################################################   
## Hand-written symbols identification system   ##
##################################################

1. Preparation:

	A. Package needed: numpy, matplotlib,torch,sklearn,torchvision,tqdm(only for train.py)
        B. For train.py, if the gpu is not available, we will set up DEVICE variable to "cpu"
        C. For test.py, we set up the device to "cpu"

2. How to Run:

	A. train.py
        	a.Locate the data and labels to the same folder with train.py, the files name need to be "Images.npy" and 			"Labels.npy"
		b.The shape of "Images.npy" need to be (N,150,150) where N is the number of dataset
  		c.The shape of "Labels.npy" need to be (N,) where N is the number of Label
                d.Please ensure the training dataset is average distributed for each class for using the stratified 				techniques to split the train and test data, if not, please uncomment the line 24 and comment the line 23
                e. Run the train.py file, the training process should start if the RAM has enough space

        B. test.py
               	a.Locate the "trained_model_2.pt" file to the same folder with the "test.py"
		b.The shape of "test_images.npy" need to be (N,150,150) where N is the number of dataset
  		c.Call the function called "predict_func" which takes a numpy array in a shape of (N,150,150). the function 			will return the predicted label as a format of numpy array (N,)
		


##################################################
The code for baseline model that we used for comparison in our paper can be found in Baseline_TrainTest.ipynb notebook

              


