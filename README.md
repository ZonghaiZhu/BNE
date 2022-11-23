# Balanced Neighbor Exploration for Semi-Supervised Node Classification on Imbalanced Graph Data

## Overview

This work proposes a Balanced Neighbor Exploration (BNE) algorithm that explores the connected neighbors of nodes in each class and computes the count matrix where each row reflects how frequently the nodes of one class explore their neighbors. Then, BNE selects nodes for each class according to the count matrix and provides a balanced training scenario. BNE exhibits a straightforward implementation whose backbone includes multiplying the normalized adjacency matrix with the original feature matrix of nodes to get the embedding features and learning a two-layer linear network. Moreover, when BNE completes computing the normalized adjacency matrix, it conducts the neighbor exploration process by using sum and sort operations, exhibiting a fast speed. Experiments validate the efficiency and effectiveness of the proposed BNE.

##
The document 'BNE' contains the code of the proposed BNE.

The documents mainly include BNE, Backbone_RNs, BNE_Reddit, and PPRGo_RN_Reddit. BNE and BNE_Reddit are our implementations, and BNE is validated on 5 data sets downloaded from torch_geometric.datasets. BNE_Reddit conducts the experiment on the Reddit data set. The rest documents are primary comparison methods ReNode (RN). Backbone_RNs, including Backbone_RN_CB, Backbone_RN_Focal, Backbone_RN_RW, mainly combines the backbone of BNE with all RN-based methods, and it is validated on 5 data sets downloaded from torch_geometric.datasets. PPRGo_RN_Reddit is the combination of PPRGo and ReNode.

For convenience, when we conduct the experiment on Reddit, we only tune the hyper-parameter on the equal-type splitting with 20 training nodes per class. When we get the best combination of hyper-parameters except for the hyper-parameter scale_w in PPRGo_RN_Reddit and explore coefficient in BNE, we use this combination of hyper-parameters to learn the model on the other kinds of splitting. This strategy is utilized for BNE_Reddit and PPRGo_RN_Reddit.

The other documents are BNE_convergence, BNE_K, BNE_Time, RN_Time, NE_acc, ProcessingTime.

BNE_convergence: present the convergence curve
BNE_K: present the training and validaiton performance with varying parametner K
BNE_Time and RN_Time: present the training time of BNE and RN.
NE_acc: the accuracy of the explored nodes by using neighbor exploration
ProcessingTime: the processing time of neighbor exploration in BNE and re-weighting in ReNode.


## Results

We have provided all classification results in the uploaded codes. Please check the csv files.

All results can be reproduced by running main.py. In main.py, you can select the dataset and adjust the parameters. The main.py will use model.py which contains the fit function and predict function. We have used the grid search to conduct the experiment and record the results on validation and test data. The parameters combination that achieves the highest results on the validation data is selected to predict the test data. 


## Dependencies

* python >= 3.7.6
* pytorch >= 1.6.0
* torch_geometric >= 1.6.1
* numpy >= 1.19.2
* scikit-learn >= 0.23.1
* scipy >= 1.41

## Run Program

* main.py calls the main function to run the program
* model.py represents the model and contains training and testing
* net.py is the structure of the used neural network
* utils.py provides some tools to assist program execution 
* load_data.py provides the partition of data and explores neighbors

Run main.py directly to get the results on validation and test data. You can change the parameters and choose different data set in argparse.ArgumentParser() to see different results. 

## Datasets

All datasets used in this work can be downloaded from torch_geometric.datasets.Planetoid and torch_geometric.datasets.Amazon automatically. Please download the Reddit data set according to "https://github.com/GraphSAINT/GraphSAINT" provided in the paper "GraphSAINT: Graph Sampling Based Inductive Learning Method". The Reddit uploaded by the paper of ReNode and PPRGo is wrong because the scale of indices in the csr_matrix is over 100 million. However, the real Reddit has about 10 million edges.

## Customization

### How to Prepare Your Own Dataset?

You can prepare the graph data with adjacency matrix A, feature matrix X, and corresponding label Y. Our BNE mainly conduct operations  on the adjacency matrix A. Change the corresponding position of A, X , and Y in the code.



