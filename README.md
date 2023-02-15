# Research Source Code

The three big files are **NetworkCreationAndMeasurements** , **PostMeasurementFINAL** and **DeepLearning**

## NetworkCreationAndMeasurements

This is where the different networks are built. You can run everything with **exportRawData** (after initialization). This will go through and preprocess the images if needed, do the image segmentation of your choice (seg_type), create the chosen network type (edge_type) with the chosen radial distance interval (R1 to R2).

## PostMeasurementFINAL

This is where the dimensionality reduction of the raw data, clustering, and precision/recall calculations are performed. 

## DeepLearning

This is where the deployment of the Convolutional Neural Network (CNN) will take place.

