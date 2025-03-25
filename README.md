## Z-GCNETs: Time Zigzags at Graph Convolutional Networks for Time Series Forecasting

### Z-GCNETs
Code for paper [Z-GCNETs: Time Zigzags at Graph Convolutional Networks for Time Series Forecasting](https://arxiv.org/abs/2105.04100) (ICML 2021).

### Requirements
Python 3.6.5, Pytorch >=1.4.0, gudhi 3.3.0, networkx 2.3.0, dionysus 2.0.8, ripser 0.4.1, and persim 0.1.1.

## Datasets 
For the zigzag persistence image (ZPI) of PeMS (i.e., PeMSD4 and PeMSD8):
1. please run the functions, i.e., zigzag_persistence_diagrams and zigzag_persistence_images in PEMS_ZPD_ZPI_generation.py
2. download ZPI (including PeMSD4_ZPI_{alpha = 0.3}, PeMSD4_ZPI_{alpha = 0.5}, and PeMSD8_ZPI_{alpha = 0.3}) at https://www.dropbox.com/sh/ry0klyksjk22mtu/AAAhi1BvfHZnpjK16tfuZmnUa?dl=0. 

After obtaining the corresponding train, validation, and test sets, please put them in the …/tda_data/tda_PEMS0X folder (X can be 4 or 8).
