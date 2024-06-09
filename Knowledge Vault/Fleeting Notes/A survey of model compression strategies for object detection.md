#### Date : 03-04-2024    Time : 9:37 am

Status : #concept

category Tag :

Tags :
# Fleeting Notes:%% begin annotations %%  
  
  

 
  
  
<mark style="background-color: #ffd400">Note</mark>  
 Firstly, these compression techniques were divided into six categories: network pruning, lightweight network design, neural architecture search (NAS), low-rank decomposition, network quantization, and Knowledge distillation (KD) methods  
  
<mark style="background-color: #ffd400">Note</mark>  
 Keywords Deep learning · Deep neural networks · Object detection · Model compression  
  
<mark style="background-color: #ffd400">Note</mark>  
 purpose of object detection is to recognize and identify the categories and locations of objects that appear in an image  
  
<mark style="background-color: #ffd400">Note</mark>  
 most state-of-the-art models exploit deep learning-based networks as their backbones to extract features  
  
  
   
  
<mark style="background-color: #ffd400">Note</mark>  
 models often need to be deployed on embedded development platforms with weak computing power, such as autonomous driving and intelligent patrol cars  
  
<mark style="background-color: #ffd400">Note</mark>  
 limitations of current complex object detection models mainly come from three aspects:1) model size, 2) memory at runtime, and 3) number of computations  
  
<mark style="background-color: #2ea8e5">Note</mark>  
 Table 1 The current DNNs  
  
<mark style="background-color: #ffd400">Note</mark>  
 top conferences and journals contained in class A and part of class B of computer vision recommended by China Computer Federation (CCF) and Association for Computing Machinery (ACM)  
  
<mark style="background-color: #f19837">Note</mark>  
 2 Model compression strategies  
  
<mark style="background-color: #ffd400">Note</mark>  
 field can be divided into the following six research directions:  
  
<mark style="background-color: #ffd400">Note</mark>  
 Fig. 1. It can be seen that network pruning and lightweight network structure design methods account for the largest proportions among the current model compression methods  
  
  
   
  
<mark style="background-color: #2ea8e5">Note</mark>  
 Fig. 1 Distribution of the research directions concerning model compression  
  
  
   
  
<mark style="background-color: #2ea8e5">Note</mark>  
 Fig. 2 The distribution of top conferences and journals papers in model compression research  
  
<mark style="background-color: #f19837">Note</mark>  
 2.1 Network pruning  
  
<mark style="background-color: #ffd400">Note</mark>  
 network pruning approach is one of the most extensive model compression methods in DNNs  
  
<mark style="background-color: #5fb236">Note</mark>  
 Its main idea is to remove redundant parameters by designing different standards for judging whether parameters are important to make the network sparse and achieve compression.  
  
  
   
  
<mark style="background-color: #2ea8e5">Note</mark>  
 Fig. 4 Classification of Network Pruning approaches  
  
<mark style="background-color: #ffd400">Note</mark>  
 network pruning methods can be divided into structured pruning and unstructured pruning  
  
<mark style="background-color: #ffd400">Note</mark>  
 As shown in Fig. 4, unstructured pruning mainly includes unstructured weight pruning  
  
<mark style="background-color: #5fb236">Note</mark>  
 unstructured weight pruning can directly remove unimportant weights from each layer to reduce the number of model parameters  
  
<mark style="background-color: #5fb236">Note</mark>  
 Structured pruning is implemented by deleting the unimportant structures contained in the whole DNN, these structures include convolution kernels, channels, filters, layers, and so on.  
  
<mark style="background-color: #f19837">Note</mark>  
 2.1.1 Unstructured pruning  
  
<mark style="background-color: #5fb236">Note</mark>  
 The principle of unstructured weight pruning is shown in Fig. 5. The unimportant weight edges of the given network are pruned while minimally affecting the overall structure  
  
<mark style="background-color: #ffd400">Note</mark>  
 Han Song team of Stanford University has produced many achievements in the field of model compression  
  
<mark style="background-color: #ff6666">Note</mark>  
 Han Song team and NVIDIA team [13] jointly first proposed a method of pruning weights whose absolute values are less than a given threshold, and they then trained a sparse network to learn the remaining connection weight. Their model achieved a high sparsity rate at the weight level  
  
  
   
  
<mark style="background-color: #2ea8e5">Note</mark>  
 Fig. 5 Weight pruning  
  
  
   
  
<mark style="background-color: #2ea8e5">Note</mark>  
 Fig. 6 The research situations of various research institutions and teams on regarding Unstructured Pruning  
  
<mark style="background-color: #ff6666">Note</mark>  
 propose a three-stage pipeline-based deep compression method, which combines pruning, training quantization and Huffman coding to reduce the storage requirements of y the examined neural network while maintaining the original accuracy  
  
<mark style="background-color: #5fb236">Note</mark>  
 reduce the communication overhead, Lin et  al. [15] proposed a deep gradient compression (DGC) method, which only selects a part of the most "important" gradient elements in each iteration for sending.  
  
<mark style="background-color: #5fb236">Note</mark>  
 method of pruning weights by providing thresholds for each layer is called layerwise magnitude-based pruning (LMP)  
  
<mark style="background-color: #5fb236">Note</mark>  
 facilitate the use of LMP by nonprofessional users, the Anhui Province Key Lab of Big Data Analysis and Application [16] proposed an approach called optimization based LMP (OLMP) to automatically adjust the pruning threshold of LMP  
  
<mark style="background-color: #5fb236">Note</mark>  
 threshold adjustment problem was transformed into a constrained optimization problem, minimizing the size of the pruned model subject to an accuracy loss constraint  
  
<mark style="background-color: #5fb236">Note</mark>  
 2021, Lee et al. [20] proposed an importance score for global pruning by studying layered sparsity, namely, the layer-adaptive magnitude-based pruning (LAMP) score, and their approach did not require any hyperparameter tuning or heavy computation.  
  
<mark style="background-color: #a28ae5">Note</mark>  
 To speed up the training speed and prevent the loss of accuracy caused by the fine-tuning or modification of the network architecture during or after training, Frankle’s team [25] at Massachusetts Institute of Technology (MIT) proposed random initialization pruning methods, which are different from magnitude-based pruning after training, randomly shuffling the pruned weights within each layer or sampling new initial values to preserve or improve accuracy.  
  
<mark style="background-color: #ffd400">Note</mark>  
 On the whole, unstructured pruning, regardless of its structural position, has less impact on the overall model accuracy and can achieve a high sparsity rate  
  
<mark style="background-color: #e56eee">Note</mark>  
 we should also pay attention to the irregular memory access caused by unstructured pruning  
  
<mark style="background-color: #ffd400">Note</mark>  
 unstructured pruning, the research of MIT Han Song team has provided many contributions.  
  
<mark style="background-color: #f19837">Note</mark>  
 2.1.2 Structured pruning  
  
<mark style="background-color: #5fb236">Note</mark>  
 concept of structural pruning was proposed based on the limitations of unstructured pruning. Structural pruning is realized by deleting the unimportant structures in a whole DNN and minimizing the reconstruction error. According to the granularity of structural pruning, structural pruning methods can be divided into neuron pruning, filter/feature map/ kernel pruning, channel pruning and layer pruning.  
  
<mark style="background-color: #f19837">Note</mark>  
 (1) Neuron Pruning  
  
<mark style="background-color: #5fb236">Note</mark>  
 (1) Neuron Pruning After a neural network receives an input, not all neurons in the network are activated, so the output value of those deactivated neurons can be set to zero through a certain criterion, which can reduce structural redundancy  
  
<mark style="background-color: #5fb236">Note</mark>  
 Neuron pruning is relevant to weight pruning.  
  
<mark style="background-color: #ffd400">Note</mark>  
 Weight pruning focuses on the pruning of weight edges, while neuron pruning focuses on the pruning of the neurons pointed out by the weight edges.  
  
  
   
  
<mark style="background-color: #2ea8e5">Note</mark>  
 Fig. 7 The research situations of various research institutions and teams regarding Structured Pruning  
  
<mark style="background-color: #ffd400">Note</mark>  
 TuSimple Team [29] introduced a new parameter-scale factor to scale the output of a specific structure to achieve data-driven and end-to-end pruning  
  
<mark style="background-color: #a28ae5">Note</mark>  
 DiDi AI research team [33] proposed an automatic structured pruning framework called AutoCompress based on the alternating direction method of multipliers (ADMM)  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Ma et al. [34] also used the ADMM algorithm to obtain a sparse structure called partial convolution (PCONV).  
  
<mark style="background-color: #ffd400">Note</mark>  
 research on neuron pruning mainly focuses on the evaluation of neuron contribution rates and importance levels, and some studies have combined these topics with structural weight pruning  
  
<mark style="background-color: #e56eee">Note</mark>  
 On the whole, few studies have examined neuron pruning.  
  
<mark style="background-color: #f19837">Note</mark>  
 (2) Filter/Feature Map/Kernel Pruning  
  
<mark style="background-color: #ffd400">Note</mark>  
 Convolutional neural networks (CNNs) usually have significant redundancy between different filters and feature maps. A feature map is an output of the network, and a filter is a network parameter.  
  
<mark style="background-color: #5fb236">Note</mark>  
 feature map is subtracted, the filter connected to it is also removed.  
  
<mark style="background-color: #ffd400">Note</mark>  
 Some differences can be found between the filter pruning method and the neuron pruning method.  
  
<mark style="background-color: #5fb236">Note</mark>  
 The filters pruning method prunes CNN at the filter rather level than in terms of a single connection (parameter) or neuron  
  
<mark style="background-color: #ffd400">Note</mark>  
 Filter pruning is the most important structured pruning method.  
  
<mark style="background-color: #f19837">Note</mark>  
 Methods based on Sparse Constraint Regularization  
  
<mark style="background-color: #ffd400">Note</mark>  
 scholars have studied network parameter regularization methods based on group sparsity to penalize unimportant parameters.  
  
<mark style="background-color: #a28ae5">Note</mark>  
 team at the University of Maryland and NEC Labs America [40] proposed a structural sparse CNN regularization and acceleration method, which can greatly reduce the cost of computation by removing all the filters and their connected feature maps in the network that have little impact on the output accuracy.  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Skolkovo Institute of Science and Technology [41] used the online balanced descent (OBD) algo-  
  
<mark style="background-color: #a28ae5">Note</mark>  
 rithm [12] to regard the convolution operation as a matrix multiplication calculation, and used the group sparsity method to achieve sparse matrix multiplication and improve the calculation speed of their approach  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Megvii Technology and Xi’an Jiaotong University [42] proposed a channel selection strategy based on the LASSO and a least-squares reconstruction algorithm to prune filters.  
  
<mark style="background-color: #ffd400">Note</mark>  
 Many methods based on the sparse learning (SL) framework have also been proven to be effective  
  
<mark style="background-color: #a28ae5">Note</mark>  
 [43–47] of Université de Sherbrooke proposed an SL framework, which allows learned and selected filters to be pruned in a CNN while considering the neuron budget  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Université de Sherbrooke proposed an SL framework, which allows learned and selected filters to be pruned in a CNN while considering the neuron budget  
  
<mark style="background-color: #ffd400">Note</mark>  
 Université de Sherbrooke proposed an SL framework, which allows learned and selected filters to be pruned in a CNN while considering the neuron budget  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Université de Sherbrooke proposed an SL framework, which allows learned and selected filters to be pruned in a CNN while considering the neuron budget  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Université de Sherbrooke proposed an SL framework, which allows learned and selected filters to be pruned in a CNN while considering the neuron budget  
  
<mark style="background-color: #ffd400">Note</mark>  
 neuron budget  
  
<mark style="background-color: #ffd400">Note</mark>  
 regularization based on pruning technique requires a sensitivity analysis for each layer and additional computations.  
  
<mark style="background-color: #ffd400">Note</mark>  
 NVIDIA [49] proposed a global rescaling method that depends on the standards of all layers and does not require sensitivity estimation.  
  
<mark style="background-color: #f19837">Note</mark>  
 Methods based on adaptive and dynamic pruning  
  
<mark style="background-color: #a28ae5">Note</mark>  
 A team at the University of Sydney and Baidu [54] proposed the learning filter pruning criterion (LFPC) by considering cross-layer filter distribution diversity, this approach adaptively selects the appropriate pruning criteria for different functional layers.  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Wang et al. [55] of the University of Tennessee proposed a layer-adaptive filter pruning approach based on structural redundancy reduction, which builds a graph for each convolutional layer of a CNN to measure the redundancy in each layer. This approach prunes unimportant filters in the most redundant layer(s) rather than the filters with the least importance across all layers.  
  
<mark style="background-color: #a28ae5">Note</mark>  
 To overcome the problem that many extra hyperparameters and training epochs are caused by the existing filter pruning methods, Ruan [57] of the Chinese Academy of Sciences proposed a novel dynamic and progressive filter pruning scheme (DPFPS) that directly learns a structured sparsity network from scratch. In particular, the DPFPS specifically imposes a new structured sparsity inducing regularization upon the expected pruning parameters in a dynamically sparse manner.  
  
<mark style="background-color: #f19837">Note</mark>  
 Methods based on limited data and without retraining or fine-tuning  
  
<mark style="background-color: #ffd400">Note</mark>  
 Considering that the current methods typically need to use the original data for retraining after pruning to compensate for the loss of accuracy, due to the data privacy problem, only a small part of the data may actually be obtained.  
  
<mark style="background-color: #e56eee">Note</mark>  
 Ding team of Tsinghua University [61] also proposed a method to safely remove redundant filters, but it is not ideal for large scale datasets because the model needs to be trained from scratch [62]  
  
<mark style="background-color: #a28ae5">Note</mark>  
 the team further proposed approximated oracle filter pruning (AOFP) [63], which continuously searches for the least important filters in a binary search manner, performs pruning attempts by randomly masking out filters, accumulates the resulting errors, and finetunes the model via a multipath framework.  
  
<mark style="background-color: #5fb236">Note</mark>  
 Most of the previous works involved "hard filter pruning", which means deleting the pruned filters directly. Therefore, this approach reduces the capacity of the original model and leads to performance degradation. In addition, after pruning the filters of the pretraining model, fine-tuning is needed.  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Most of the previous works involved "hard filter pruning", which means deleting the pruned filters directly. Therefore, this approach reduces the capacity of the original model and leads to performance degradation. In addition, after pruning the filters of the pretraining model, fine-tuning is needed.  
  
<mark style="background-color: #f19837">Note</mark>  
 Others  
  
<mark style="background-color: #ffd400">Note</mark>  
 scholars have studied pruning methods that can directly specify the accuracy of the model after pruning.  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Pravendra Singh team at the IIT Kanpur Institute [67] proposed a play and prune (PP) framework consisting of an auto-balanced filter pruning (AFP) [51] module and a pruning rate controller (PRC) module  
  
<mark style="background-color: #a28ae5">Note</mark>  
 This method allows for directly specifying the desired error tolerance instead of conducting this step at the pruning level.  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Luo Jianhao’s team at the National Key Software Technology Laboratory of Nanjing University [69] proposed a Thinet architecture based on filter pruning by using a greedy  
  
<mark style="background-color: #ff6666">Note</mark>  
 Luo Jianhao’s team at the National Key Software Technology Laboratory of Nanjing University [69] proposed a Thinet architecture based on filter pruning by using a greedy  
  
<mark style="background-color: #a28ae5">Note</mark>  
 algorithm. They considered filter pruning to be an optimization question and pointed out that whether a filter can be pruned depends on the input of the next layer rather than that of the current layer, they finally pruned the filter corresponding to the layer with the weakest input channel data in the next layer. This method has also been one of the main comparative approaches in follow-up research  
  
<mark style="background-color: #ff6666">Note</mark>  
 algorithm. They considered filter pruning to be an optimization question and pointed out that whether a filter can be pruned depends on the input of the next layer rather than that of the current layer, they finally pruned the filter corresponding to the layer with the weakest input channel data in the next layer. This method has also been one of the main comparative approaches in follow-up research  
  
<mark style="background-color: #a28ae5">Note</mark>  
 The Pravendra Singh team at the IIT Kanpur Institute [74] proposed a new filter structure called hetconv based on heterogeneous kernel to reduce the number of FLOPs without incurring the loss of accuracy.  
  
<mark style="background-color: #ff6666">Note</mark>  
 filter, feature map, and kernel pruning method, it can be seen that the main techniques used in the research on filter pruning methods are the sparse constraint regularization method to penalize unimportant parameters, and cross-layer correlation, importance ranking and other mechanisms to evaluate importance and remove redundancy, and finally obtaining a sparse structure. In terms of research purposes, some studies mainly focus on filter pruning without fine-tuning and retraining, and pruning in the case of limited data. Some scholars specialize in adaptive and dynamic pruning  
  
<mark style="background-color: #5fb236">Note</mark>  
 Generally, filter pruning methods are relatively excellent, and the global dynamic pruning may be the future trend.  
  
<mark style="background-color: #f19837">Note</mark>  
 (3) Channel Pruning  
  
<mark style="background-color: #ffd400">Note</mark>  
 filter pruning and channel pruning approaches are coarsegrained pruning methods, and their pruning techniques are similar  
  
<mark style="background-color: #5fb236">Note</mark>  
 channel pruning can solve the dimensional mismatch problem caused by filter pruning. The main idea of channel pruning is to directly delete entire channels from the convolutional layer of a network. After pruning, the model can directly be trained to obtain higher compression ratio and a shorter computing time  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Ding’s team [78] at Tsinghua University proposed ResRep, a novel method for lossless channel pruning, which slims down a CNN by reducing the widths (numbers of output channels) of convolutional layers.  
  
<mark style="background-color: #a28ae5">Note</mark>  
 In 2021, Huawei Noah’s Ark Lab worked with Xiamen University [85] to propose a collaborative compression (CC) scheme, which joins channel pruning and tensor decomposition to compress CNN models by simultaneously learning the model sparsity and low-rank status  
  
<mark style="background-color: #a28ae5">Note</mark>  
 To reduces the risk of pruning important channels prematurely, another team including Hou et al. [91] propose a novel Channel Exploration methodology (CHEX) in 2022, which repeatedly prune and regrow the channels throughout the training process. They tackle the channel pruning problem via a well known column subset selection (CSS) formulation. The results show that compressed ResNet-50 model on ImageNet dataset achieves 76%  
  
<mark style="background-color: #a28ae5">Note</mark>  
 top1 accuracy with only 25% FLOPs of the original ResNet-50 model, outperforming previous state-of-the-art channel pruning methods.  
  
<mark style="background-color: #5fb236">Note</mark>  
 Through the research on channel pruning methods, it can be seen that among these approaches, L1 regularization using the scaling factor of the BN layer, sensitivity and coupling the correlations of the channels in each layer to remove redundancy are most popular.  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Most recently, a team of National Yang Ming Chiao Tung University proposes an accurate and efficient object detector capable of performing real-time inference under the hardware constraints of an edge device by leveraging structural pruning, feature distillation, and NAS [92].  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Most recently, a team of National Yang Ming Chiao Tung University proposes an accurate and efficient object detector capable of performing real-time inference under the hardware constraints of an edge device by leveraging structural pruning, feature distillation, and NAS [92].  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Most recently, a team of National Yang Ming Chiao Tung University proposes an accurate and efficient object detector capable of performing real-time inference under the hardware constraints of an edge device by leveraging structural pruning, feature distillation, and NAS [92].  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Most recently, a team of National Yang Ming Chiao Tung University proposes an accurate and efficient object detector capable of performing real-time inference under the hardware constraints of an edge device by leveraging structural pruning, feature distillation, and NAS [92].  
  
<mark style="background-color: #ffd400">Note</mark>  
 Generally, the channel pruning methods have good compression effect and do not produce a sparse matrix, so special software is not required.  
  
<mark style="background-color: #f19837">Note</mark>  
 (4) Layer Pruning  
  
<mark style="background-color: #5fb236">Note</mark>  
 Layer pruning method is one of the deepest pruning methods, which directly prunes a layer in the network. The degree of compression of layer pruning is great, but the accuracy is relatively lower.  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Yang et al. [94] in MIT used the method of energy perception to calculate the energy consumption of each layer, preferentially pruning the layers with high energy consumption. In addition, to prevent incorrect pruning, the weight that resulted in the greatest decrease in accuracy after pruning was retained. Subsequently, the team cooperated with Google [95] to propose Netadapt, which also takes hardware metrics (delay, energy consumption, etc.) as pruning evaluation criteria, uses empirical metrics for evaluation, and automatically iteratively prunes the pretraining network on a mobile platform until the imposed resource budget is met.  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Yang et al. [94] in MIT used the method of energy perception to calculate the energy consumption of each layer, preferentially pruning the layers with high energy consumption. In addition, to prevent incorrect pruning, the weight that resulted in the greatest decrease in accuracy after pruning was retained. Subsequently, the team cooperated with Google [95] to propose Netadapt, which also takes hardware metrics (delay, energy consumption, etc.) as pruning evaluation criteria, uses empirical metrics for evaluation, and automatically iteratively prunes the pretraining network on a mobile platform until the imposed resource budget is met.  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Yang et al. [94] in MIT used the method of energy perception to calculate the energy consumption of each layer, preferentially pruning the layers with high energy consumption. In addition, to prevent incorrect pruning, the weight that resulted in the greatest decrease in accuracy after pruning was retained. Subsequently, the team cooperated with Google [95] to propose Netadapt, which also takes hardware metrics (delay, energy consumption, etc.) as pruning evaluation criteria, uses empirical metrics for evaluation, and automatically iteratively prunes the pretraining network on a mobile platform until the imposed resource budget is met.  
  
<mark style="background-color: #a28ae5">Note</mark>  
 In the application of object detection, the Guangdong Key Laboratory of Modern Control Technology uses channel pruning and layer pruning to compress the YOLOv3-SPP garbage detection model, and the compressed model is directly deployed on the low-computing edge computing device carried by the robot to automatically classify and recycle garbage[279].  
  
<mark style="background-color: #a28ae5">Note</mark>  
 Google Brain and DeepMind [96] proposed a method to train sparse neural networks with a fixed parameter count and a fixed computational cost throughout training. This approach updates the topology of a sparse network during training by using parameter magnitudes and infrequent gradient calculations, this approach requires fewer FLOPs to achieve a given level of accuracy.  
  
<mark style="background-color: #5fb236">Note</mark>  
 Through a large number of experiments, Le et al. [98] of VinAI Research studied random pruning and learning rate restarting, and proposed that compared to traditional finetuning, in general, learning rate restarting is an efficient way to retrain pruned networks to recover performance drops caused by pruning.  
  
<mark style="background-color: #f19837">Note</mark>  
 Discussion  
  
<mark style="background-color: #5fb236">Note</mark>  
 According to whether the network structure is changed after pruning, network pruning can be divided into structured pruning and unstructured pruning  
  
<mark style="background-color: #ffd400">Note</mark>  
 Unstructured pruning (weight pruning) does not change the original structure of the network, but the resulting sparse network needs to be supported by special hardware libraries or computing libraries  
  
<mark style="background-color: #5fb236">Note</mark>  
 In contrast, structured pruning directly removes part of the network structure and is more suitable for general hardware acceleration.  
  
<mark style="background-color: #5fb236">Note</mark>  
 According to the granularity of pruning, structure pruning can be further divided into neuron pruning, kernel pruning, channel pruning and layer pruning.  
  
<mark style="background-color: #ffd400">Note</mark>  
 Neuron pruning is based on a single neuron, which is suitable for any type of layer, but the pruning effect is poor  
  
<mark style="background-color: #ffd400">Note</mark>  
 Kernel pruning is based on convolution kernels and is usually applied to convolutional layers;  
  
<mark style="background-color: #ffd400">Note</mark>  
 Channel pruning is an extension of kernel pruning, which prunes multiple channels in a convolution kernel.  
  
<mark style="background-color: #ffd400">Note</mark>  
 Layer pruning is one of the most extreme structured pruning techniques applicable to some deeper neural networks where some layers can be considered useless or redundant.  
  
<mark style="background-color: #f19837">Note</mark>  
 2.2 Lightweight network design  
  
  
%% end annotations %%


---
## References:
[!Cite]
[1]

Z. Lyu _et al._, “A survey of model compression strategies for object detection,” _Multimed Tools Appl_, Nov. 2023, doi: [10.1007/s11042-023-17192-x](https://doi.org/10.1007/s11042-023-17192-x). Available: [https://link.springer.com/10.1007/s11042-023-17192-x](https://link.springer.com/10.1007/s11042-023-17192-x). [Accessed: Mar. 14, 2024]

%% Import Date: 2024-04-03T09:37:39.167+05:30 %%
