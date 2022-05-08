# Msc Computer Vision, Machine Learning and Robotics Thesis Github Repository
### Title: A Lightweight Supervised Intrusion Detection System for Future Smart Grid Metering Network
Abstract: The integration of information and communication technologies into the power generation, 
transmission and distribution system provides a new concept called Smart Grid (SG). The 
wide variety of devices connected to the SG communication infrastructure generates heterogeneous data with different Quality of Service (QoS) requirements and communication 
technologies. Hence, this project aims to design a robust IDS dealing with anomaly SG data and impose the proper defence algorithm to alert the network. An intrusion Detection 
System (IDS) is a surveillance system monitoring the traffic flow over the network, seeking any abnormal behaviour to detect possible intrusions or attacks against the SG system.

### Installation 
It is highly recommended to use Colaboratory ([Colab](https://colab.research.google.com/notebooks/welcome.ipynb)) to run the notebooks, because it allows to write and execute Python code in a browser with:

- Zero configuration required
- Free access to GPUs and TPUs
- Most libraries pre-installed
- Only one requirement, a google account
- Most common Machine Learning frameworks pre-installed and ready to use

### Usage 
  #### Prepare Data
  > Note: Preprocessed data can be found in the PreprocessedData folder
  
  In order to prepare your data follow the steps below:

  1. Download the [data_preprocessing.ipynb](https://github.com/sotirischatzimiltis/MscThesis/blob/main/DataPreProcessing/data_preprocessing.ipynb)

  2. Download the NSLKDD dataset: [NSLKDD dataset](https://github.com/sotirischatzimiltis/MscThesis/tree/main/NSLKDD)
    
  3. Open [Colab](https://colab.research.google.com/notebooks/welcome.ipynb) and sign in to your Google account. If you do not have a Google account, you can create one [here](https://accounts.google.com/signup/v2/webcreateaccount?hl=en&flowName=GlifWebSignIn&flowEntry=SignUp).

  4. Go to _File > Upload notebook > Choose file_ and browse to find the downloaded notebook [data_preprocessing.ipynb](https://github.com/sotirischatzimiltis/MscThesis/blob/main/DataPreProcessing/data_preprocessing.ipynb) If you have already uploaded the notebook to Colab you can open it with _File > Open notebook_ and choose **data_preprocessing.ipynb**. 
  
  #### Spiking Neural Network
  In order to train a SNN model follow the steps below:




