# Training from Scratch using CIFAR100

- The program implements to classify the CIFAR100 Dataset with the Convolution Neural Network.
- The customized CNN in this code has 6 convolution layers and with different learning rates for different convolution layers in the network.
- The following are the parameters used CNN network model :
    - Images input size = (64, 64, 3)
    - Optimizer = Adam
    - Batch size = 128
    - Epochs = 100
    - Kernel size = (3, 3)


## Dataset
The dataset used for this code is ["CIFAR100 Dataset"](https://www.cs.toronto.edu/~kriz/cifar.html) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
 - The total images in the dataset is 60,000.
 - The dataset has 100 classes. 
 - The dataset containing 600 images per class.
 - There are 500 training images and 100 testing images per class.


## Results
The CNN (training from Scratch) with CIFAR100 DataSet is made for a 3 runs and the accuracies are recorded as follows:

### First run:
In the first run, for 100 epochs:
 - Training accuracy = 73.09 %
 - Testing accuracy = 42.65 %
 - Time Elapsed = 894.179 secs

### Second run:
In the second run, for 100 epochs:
 - Training accuracy = 76.07 %
 - Testing accuracy = 42.56 %
 - Time Elapsed = 892.103 secs

### Third run:
In the third run, for 100 epochs:
 - Training accuracy = 78.38 %
 - Testing accuracy = 43.96 %
 - Time Elapsed = 888.460 secs

### Average Accuracies:
 - The average training accuracy for 3 runs : 78.84 %
 - The average testing accuracy for 3 runs : 43.65%
 - The average time for 3 runs : 891.58 secs

## Configuration Instructions
The [Project](https://github.com/sowmi06/Naive-Bayes-N-estimates.git) requires the following tools and libraries to run the source code.
### System Requirements 
- [Anaconda Navigator](https://docs.anaconda.com/anaconda/navigator/install/)
    - Python version 3.6.0 – 3.9.0
 
- Python IDE (to run ".py" file)
    - [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows), [Spyder](https://www.psych.mcgill.ca/labs/mogillab/anaconda2/lib/python2.7/site-packages/spyder/doc/installation.html) or [VS code](https://code.visualstudio.com/download)

### Tools and Library Requirements 
    
- [Numpy](https://numpy.org/install/)
  
- [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)

- [TensorFlow](https://www.tensorflow.org/install/pip)


## Operating Instructions

The following are the steps to replicate the exact results acquired from the project:

- Satisify all the system and the tool, libraries requirements.
- Clone the [Training-from-Scratch-using-CIFAR100](https://github.com/sowmi06/Training-from-Scratch-using-CIFAR100.git) repository into your local machine. 
- The [Training_from_Scratch.py](https://github.com/sowmi06/Training-from-Scratch-using-CIFAR100/blob/main/Training_from_Scratch.py) has the code for the preprocessing steps and final classifiaction output.
