# -*- coding: utf-8 -*-

import pandas as pd
import sklearn.preprocessing as skp
import sklearn.decomposition as skd
import sklearn.cluster as skc
import numpy as np

class Clustering():
    '''
    Do this class with batches of trials... parallelize over hyperparameter iterations
    '''
    def __init__(self, sampledImagesList):
        self.sampledImages = sampledImagesList
        
    def __str__(self):
        return 'Clustering Class'
    
    ## scale the data per axis
    def standardizeData(self, edge, seg, R1, R2):
        # import the data
        # TODO: make sure filepath matches with Wendian directories
        data = pd.read_csv(f'{str(R1)} to {str(R2)} - {edge} - {seg}.csv', header=None)
        
        # create instance of scalar and scale by axis
        scaler = skp.StandardScaler()
        standardData = scaler.fit_transform(data)
        return standardData
    
    ## do PCA
    def PCA(self, standardData, variance=0.95):
        #create instance of model
        pca = skd.PCA(variance)
        
        #transform data and get reduced number of components
        transformedData = pca.fit_transform(standardData)
        return transformedData
        
    ## do clustering
    def clusterData(self, transformedData, distanceThreshold=15):
        # create instance of model
        model = skc.AgglomerativeClustering(distance_threshold=distanceThreshold, n_clusters=None)
        
        # get clusters from transformed data
        clustering = model.fit_predict(transformedData)
        
        # get list of clusters and number of images in each
        clusters, counts = np.unique(clustering, return_counts=True)
        
        return clustering, clusters, counts
    
    ## make list of images in each cluster
    def indexClustersWithName(self, clustering, clusters):
        clustering = np.array(clustering)
        assignments = {}
        for i in clusters:
            indicies = np.where(clustering == i)[0]
            images = [self.sampledImages[idx] for idx in indicies]
            assignments[i] = images
            
        return assignments
    
    def exportClustering(self, edge, seg, R1, R2):
        standardData = self.standardizeData(edge, seg, R1, R2)
        transformedData = self.PCA(standardData)
        clustering, clusters, counts = self.clusterData(transformedData)
        assignments = self.indexClustersWithName(clustering, clusters)
        
        maxSize = max([len(i) for i in list(assignments.values())])
        for cluster in assignments:
            while len(assignments[cluster]) < maxSize:
                assignments[cluster].append(np.nan)
        
        pd.DataFrame(assignments).to_csv(f'Cluster Assignments - {str(R1)} to {str(R2)} - {edge} - {seg}.csv')
        
        with open(f'Cluster Sizes - {str(R1)} to {str(R2)} - {edge} - {seg}.csv', 'w') as f:
            f.write('Cluster,Count\n')
            f.writelines(f'{str(clusters[i])},{str(counts[i])}\n' for i in range(len(clusters)))
            f.close()
            
            
        
        
        
        
