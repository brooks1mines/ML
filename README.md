# Thesis Source Code

Hey there! This repo contains the basics of the code that I used for my work. The two big files are **NetworkCreationAndMeasurements** and **PostMeasurementFINAL**. I know that my code is a mess, but I think it should be mostly readable. 

## NetworkCreationAndMeasurements

This is where the different networks are built. You can run everything with **exportRawData** (after initialization). This will go through and preprocess the images if needed, do the image segmentation of your choice (seg_type), create the chosen network type (edge_type) with the chosen radial distance interval (R1 to R2).

## PostMeasurementFINAL

This is where the dimensionality reduction of the raw data, clustering, and precision/recall calculations are performed. 

Note: **Clustering** and **CalculatePR** are just **PostMeasurementFINAL** split into two files. I had to do that for some parallelization stuff. Reach out for questions.

Keep in mind that I wrote this a long time ago when I was teaching myself Python, so there are definitely many places for improvement. I would think about parallelizing everything, and rethinking some of the data structures that I used.
