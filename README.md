## Msc Computer Vision, Machine Learning and Robotics Thesis Github Repository
### Title: A Lightweight Supervised Intrusion Detection System for Future Smart Grid Metering Network
#### Abstract:
The integration of information and communication technologies into the power generation, 
transmission and distribution system provides a new concept called Smart Grid (SG). The 
wide variety of devices connected to the SG communication infrastructure generates heterogeneous data with different Quality of Service (QoS) requirements and communication 
technologies. Hence, this project aims to design a robust IDS dealing with anomaly SG data and impose the proper defence algorithm to alert the network. An intrusion Detection 
System (IDS) is a surveillance system monitoring the traffic flow over the network, seeking any abnormal behaviour to detect possible intrusions or attacks against the SG system.

####  Smart Meter Intrusion Detection System Architecture
![plot](https://github.com/sotirischatzimiltis/MscThesis/blob/main/Images/smart_meter_architecture.png)

Data traffic is captured, and passed through the pre-processing stage, whre is normalized using the min-max normalization technique.
The second pre-processing step is to encode non-numeric using one-hot encoding. 
The final pre-processing step is either feature selection or dimensionality reduction. 
The next stage is to classify if the data traffic captured as either normal or malicious, by using Spiking Neural Networks (SNN).
Last step is the evaluation of the classfier.

#### About Spiking Neural Networks
![plot](https://github.com/sotirischatzimiltis/MscThesis/blob/main/Images/leaky_neuron_SNN.JPG)

SNNs are a kind of artificial neural networks, with the difference being that rather than having the traditional artificial neuron (e.g. McCulloch-Pits) they trade it for a spiking neuron. Spiking neurons, produce a weighted sum of inputs but instead of forwarding the result into an activation function (e.g. sigmoid, ReLU), this sum contributes to the membrane potential *U(t)* of the neuron. The main condition is when *U(t)* passes a pre-defined threshold the neuron will emit a spike to successive connections. Figure above illustrates the architecture of a single spiking neuron. From the left image we see the implicit recurrence (i.e the decay part) and V (explicit recurrence) is the multiplication of Sout[t] and -θ. The right image shows an unrolled iteration of how the neuron operates. 


##### Reference 
*Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu “Training Spiking Neural Networks Using Lessons From Deep Learning”. arXiv preprint arXiv:2109.12894, September 2021.*

### Installation 
It is highly recommended to use Colaboratory ([Colab](https://colab.research.google.com/notebooks/welcome.ipynb)) to run the notebooks, because it allows to write and execute Python code in a browser with:

- Zero configuration required
- Free access to GPUs and TPUs
- Most libraries pre-installed
- Only one requirement, a google account
- Most common Machine Learning frameworks pre-installed and ready to use

> Note: if you are not going to use Google Colab you will need to make sure that you satisfy the below requirements

#### Requirements
- SNNtorch (>= 0.5.1)
- PyTorch (>= 1.11.0)
- Numpy (>= 1.21.6)
- Pandas (>= 1.3.5)
- Seaborn (>= 0.11.2)
- Matplotlib (>= 3.2.2)
- Sklearn (>= 1.0.2)
- Flwr (== 0.19.0)
- Openml (== 0.12.2)

### Usage 
  #### Prepare Data
  In order to prepare your data follow the steps below:

  1. Download one of the following scripts depending on the desired experiment
[binary_classification_std_scaler](https://github.com/sotirischatzimiltis/MscThesis/blob/main/DataPreProcessing/BinaryClassification/nsl_kdd_preprocessing_binary_stdscaler.ipynb),
[binary_classification_minmax_scaler](https://github.com/sotirischatzimiltis/MscThesis/blob/main/DataPreProcessing/BinaryClassification/nsl_kdd_preprocessing_binary_minmaxscaler.ipynb),
[multiclass_classification_std_scaler](https://github.com/sotirischatzimiltis/MscThesis/blob/main/DataPreProcessing/MultiClassClassification/nsl_kdd_preprocessing_multiclass_stdscaler.ipynb),
[multiclass_classification_minmax_scaler](https://github.com/sotirischatzimiltis/MscThesis/blob/main/DataPreProcessing/MultiClassClassification/nsl_kdd_preprocessing_multiclass_minmaxscaler.ipynb)
  > Note: Alternatively launch the desired script using the launch button 

  2. If you want to process the NSLKDD dataset in a different way you can download it from [here](https://www.unb.ca/cic/datasets/nsl.html)
    
  3. Open [Colab](https://colab.research.google.com/notebooks/welcome.ipynb) and sign in to your Google account. If you do not have a Google account, you can create one [here](https://accounts.google.com/signup/v2/webcreateaccount?hl=en&flowName=GlifWebSignIn&flowEntry=SignUp).

  4. Go to _File > Upload notebook > Choose file_ and browse to find the downloaded notebook. If you have already uploaded the notebook to Colab you can open it with _File > Open notebook_ and choose the desired notebook. 
  
#### Spiking Neural Network
In order to train a SNN model follow the steps below:

1. Download the [spiking_neural_network.ipynb](https://github.com/sotirischatzimiltis/MscThesis/blob/main/SpikingNeuralNetwork/spiking_neural_network.ipynb ).
> Note: Alternatively launch the **spiking_neural_network.ipynb** through the launch button
2. Open [Colab](https://colab.research.google.com/notebooks/welcome.ipynb) and sign in to your Google account. If you do not have a Google account, you can create one [here](https://accounts.google.com/signup/v2/webcreateaccount?hl=en&flowName=GlifWebSignIn&flowEntry=SignUp).

3. Go to _File > Upload notebook > Choose file_ and browse to find the downloaded notebook file [spiking_neural_network.ipynb](https://github.com/sotirischatzimiltis/MscThesis/blob/main/SpikingNeuralNetwork/spiking_neural_network.ipynb ). If you have already uploaded the notebook to Colab you can open it with _File > Open notebook_ and choose **spkiking_neural_network.ipynb**. 

3. Once the notebook is loaded, go to _Runtime > Change runtime type_ and from the dropdown menu, under **Hardware accelerator**, choose **GPU** and click **Save**.

5. Now you can begin the experiments. All you have to do is to upload the dataset you want and set the parameters in the cell under **Datasets** section.

6. To train the model go to _Runtime > Run all_ or click on the first cell and use **Shift + Enter** to execute each cell one by one.

7. The hyper parameters of the model can be modified in the cell under **Set Train Arguments** section.

##### Set Train Arguments
1. bsize: Batch Size
  > Default: 64
2. nhidden: Number of hidden nodes
  > Default: 4000
3. nsteps: Number of input time steps
  > Default: 25
4. b: beta/decay factor of membrane potential 
  > Default: 0.9
5. learning_rate: Learninig Rate of optimizer
  > Default: 5e-4
6. nepochs: Number of training epochs
  > Default: 10 
 
 #### Traditional ML techniques
 1. Download either the [binary](https://github.com/sotirischatzimiltis/MscThesis/blob/main/TraditionalML/traditionalml_binary_classification.py) or [multiclass](https://github.com/sotirischatzimiltis/MscThesis/blob/main/TraditionalML/traditionalml_multiclass_classification.py) classification python script.

 2. Put the correct paths to the test and train datasets.
 
 3. Execute the script.
  > Note: No need to assign values to hyperparameters. The script uses gridsearchCV using two-fold cross validation to find the best hyperparameters from a given list 
 
 #### Federated Learning
 > Note: For this part it is better to follow the documentation provided by Flower [here](https://flower.dev/docs/)
