# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:13:09 2022

@author: khloe
"""

import dask.dataframe as ddf
import sklearn.preprocessing as skp
import sklearn.decomposition as skd
import sklearn.cluster as skc
import numpy as np
import pandas as pd

class PostMeasurement():
    def __init__(self, sampledImagesList, random=False):
        self.sampledImages = sampledImagesList
        self.descriptorTypes = ['All Formalist Descriptors', 'All Critical Descriptors',
                                'Abstraction Level', 'Genres', 'Descriptors', 'Color',
                                'Line Form and Value', 'Texture', 'Shape and Space']
        self.random=random
        
    def __str__(self):
        return 'Post Measurment Class'
    
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
            
    ## get descriptors of images
    def getClusterDescriptors(self, assignments, descriptorType, splitByGenre=False):
        '''

        Parameters
        ----------
        assignments : dict {cluster: [images]}
            This dict contains the images (filename) within each cluster.
        descriptorType : str
            Type of descriptor we are looking at.
            Options:
                1. Formalist
                2. Abstraction
                3. Genre
                4. Genre Specific

        Returns
        -------
        descriptors : dict {cluster: {descriptor: count}} OR {cluster: {genre: {descriptor: count}}}
            This dict contains all descriptors (of the type of choice) per cluster 
            and their counts.

        '''
        
        # TODO: make sure that descriptor files match this format
        # TODO: convert numbers for descriptors into strings in new file used here
        #   can leave 'filler' space blank
        # TODO: insert random, randomshapes, and uniform descriptors into each descriptor file
        #   - genre = Random, RandomShapes, Uniform
        
        # TODO: make sure filepath matches with Wendian directories
        
        if descriptorType == 'All Formalist Descriptors' or descriptorType=='All Critical Descriptors':
            fp = ''
        else:
            fp = ' FINAL'
            
        if self.random == False:
            filepath = '../'
        else:
            filepath = ''
        
        allDescriptors = pd.read_csv(f'Movements/New Descriptors/Final Descriptors/{descriptorType}{fp}.csv', dtype=str).set_index('Name')
        sampleDescriptors = allDescriptors.loc[self.sampledImages]
        
        if splitByGenre == False:
            descriptors = {}
            for cluster in assignments:
                descriptors[cluster] = {}
                for image in assignments[cluster]:
                    if type(image) != float:
                        imageDescription = list(sampleDescriptors.loc[image])
                        for word in imageDescription:
                            if type(word) != float:
                                if word not in list(descriptors[cluster].keys()):
                                    descriptors[cluster][word] = 1
                                else:
                                    descriptors[cluster][word] += 1
        elif splitByGenre == True:
            # TODO: make sure filepath matches with Wendian directories
            genreDescriptors = ddf.read_csv('Genre Descriptors.csv', dtype=str).set_index('Name')
            sampleGenres = genreDescriptors.loc[self.sampledImages, :].compute()
            descriptors = {}
            for cluster in assignments: # for each cluster
                descriptors[cluster] = {} # keys = genres
                for image in assignments[cluster]: # for each image in the cluster
                    genre = sampleGenres.loc[image, 'Genre'] # get the image's genre
                    if genre not in list(descriptors[cluster].keys()): # if that genre doesn't exist in the keys
                        descriptors[cluster][genre] = {} # keys = descriptors
                        imageDescription = list(sampleDescriptors.loc[image, :]) # get descriptors (of this type) for the image
                        for word in imageDescription: # for each descriptor
                            if word != np.nan: # don't include filler space
                                if word not in list(descriptors[cluster][genre].keys()): # if doesn't exist in keys
                                    descriptors[cluster][genre][word] = 1 #create
                                else:
                                    descriptors[cluster][genre][word] += 1 # otherwise, count
                    else:
                        imageDescription = list(sampleDescriptors.loc[image, :])
                        for word in imageDescription:
                            if word != np.nan: # don't include 'filler' space
                                if word not in list(descriptors[cluster][genre].keys()):
                                    descriptors[cluster][genre][word] = 1
                                else:
                                    descriptors[cluster][genre][word] += 1
        
        else:
            raise ValueError('Invalid splitByGenre choice.')
        
        return descriptors, splitByGenre
    
    ## get significant descriptors for the clusters
    def getSpecificSignificantDescriptors(self, descriptors, descriptorType, splitByGenre=False):
        if splitByGenre == False:
            significant = {}
            for cluster in descriptors:
                words = list(descriptors[cluster].keys())
                counts = list(descriptors[cluster].values())
                
                # if cluster == 'Random':
                #     print(words)
                
                new_words = []
                new_counts = []
                for i in range(len(words)):
                    if type(words[i]) != float:
                        new_words.append(words[i])
                        new_counts.append(counts[i])
                        
                words = new_words
                counts = new_counts
                
                if len(words) > 1:
                    mean = np.mean(counts)
                    std = np.std(counts)
                
                    sigma = np.array([(i - mean) / std for i in counts]) # calculate significance
                    #print(descriptorType, sigma)
                    specific = np.where(sigma > 2.0)[0] # find idx of significant words
                    specificWords = list(map(words.__getitem__, specific)) # get those significant words
                else:
                    print(words)
                    specificWords = words
                significant[cluster] = specificWords
        elif splitByGenre == True:
            significant = {}
            for cluster in descriptors:
                significant[cluster] = {}
                for genre in descriptors[cluster]:
                    words = list(descriptors[cluster][genre].keys())
                    counts = list(descriptors[cluster][genre].values())
                    
                    mean = np.mean(counts)
                    std = np.std(counts)
                    
                    sigma = np.array([(i - mean) / std for i in counts])
                    specific = np.where(sigma > 2.0)[0]
                    specificWords = list(map(words.__getitem__, specific))
                    significant[cluster][genre] = specificWords
        else:
            raise ValueError('Invalid splitByGenre choice.')
                    
        return significant
    
    def getTrueAndFalsePositives(self, descriptors, significant, splitByGenre=False):
        truePositives = {}
        falsePositives = {}
        if splitByGenre == False:
            for cluster in descriptors:
                countTP = 0
                countFP = 0
                sigmas = significant[cluster]
                for word in descriptors[cluster]:
                    if word in sigmas:
                        countTP += descriptors[cluster][word]
                    else:
                        countFP += descriptors[cluster][word]
                truePositives[cluster] = countTP
                falsePositives[cluster] = countFP
        elif splitByGenre == True:
            for cluster in descriptors:
                truePositives[cluster] = {}
                falsePositives[cluster] = {}
                for genre in descriptors[cluster]:
                    countTP = 0
                    countFP = 0
                    sigmas = significant[cluster][genre]
                    for word in descriptors[cluster][genre]:
                        if word in sigmas:
                            countTP += descriptors[cluster][genre][word]
                        else:
                            countFP += descriptors[cluster][genre][word]
                    truePositives[cluster][genre] = countTP
                    falsePositives[cluster][genre] = countFP
        else:
            raise ValueError('Invalid splitByGenre choice.')
            
        return truePositives, falsePositives
    
    def getFalseNegatives(self, descriptors, significant, splitByGenre=False):
        falseNegatives = {}
        if splitByGenre == False:
            for cluster in descriptors:
                count = 0
                sigmas = significant[cluster]
                for otherCluster in descriptors:
                    if cluster != otherCluster:
                        otherSigmas = significant[otherCluster]
                        otherWords = descriptors[otherCluster]
                        for word in sigmas:
                            if word in otherWords and word not in otherSigmas:
                                count += descriptors[otherCluster][word]
                falseNegatives[cluster] = count
        elif splitByGenre == True:
            for cluster in descriptors:
                falseNegatives[cluster] = {}
                for genre in descriptors[cluster]:
                    count = 0
                    sigmas = significant[cluster][genre]
                    for otherCluster in descriptors:
                        if genre in list(descriptors[otherCluster].keys()):
                            otherSigmas = significant[otherCluster][genre]
                            otherWords = descriptors[otherCluster][genre]
                            for word in sigmas:
                                if word in otherWords and word not in otherSigmas:
                                    count += descriptors[otherCluster][genre][word]
                    falseNegatives[cluster][genre] = count
        else:
            raise ValueError('Invalid splitByGenre choice.')
            
        return falseNegatives
                   
    ## get precisions and recalls
    def getPrecisionAndRecalls(self, truePositives, falsePositives, falseNegatives, splitByGenre=False):
        precisions = {}
        recalls = {}
        if splitByGenre == False:
            for cluster in truePositives:
                TP = truePositives[cluster]
                FP = falsePositives[cluster]
                FN = falseNegatives[cluster]
                
                if TP == 0:
                    P = 0
                    R = 0
                else:
                    P = TP / (TP + FP)
                    R = TP / (TP + FN)
                
                precisions[cluster] = P
                recalls[cluster] = R
        elif splitByGenre == True:
            for cluster in truePositives:
                precisions[cluster] = {}
                for genre in truePositives[cluster]:
                    TP = truePositives[cluster][genre]
                    FP = falsePositives[cluster][genre]
                    FN = falseNegatives[cluster][genre]
                    
                    if TP == 0:
                        P = 0
                        R = 0
                    else:
                        P = TP / (TP + FP)
                        R = TP / (TP + FN)
                    
                    precisions[cluster][genre] = P
                    recalls[cluster][genre] = R
        else:
            raise ValueError('Invalid splitByGenre Choice.')
            
        return precisions, recalls
    
    # get overall (weighted) P&R
    def getWeightedPandR(self, precisions, recalls, clusterCounts, splitByGenre=False):
        if splitByGenre == False:
            weightedP = 0
            weightedR = 0
            for cluster in precisions:
                weightedP += clusterCounts[cluster] * precisions[cluster]
                weightedR += clusterCounts[cluster] * recalls[cluster]
        elif splitByGenre == True:
            weightedP = {}
            weightedR = {}
            for cluster in precisions:
                for genre in precisions[cluster]:
                    if genre not in list(weightedP.keys()):
                        weightedP[genre] = clusterCounts[cluster] * precisions[cluster][genre]
                        weightedR[genre] = clusterCounts[cluster] * recalls[cluster][genre]
                    else:
                        weightedP[genre] += clusterCounts[cluster] * precisions[cluster][genre]
                        weightedR[genre] += clusterCounts[cluster] * recalls[cluster][genre]
        else:
            raise ValueError('Invalud splitByGenre choice.')
        
        return weightedP, weightedR
    
    def exportClusterData(self, edge, seg, R1, R2):
        standardData = self.standardizeData(edge, seg, R1, R2)
        transformedData = self.PCA(standardData)
        clustering, clusters, counts = self.clusterData(transformedData)
        assignments = self.indexClustersWithName(clustering, clusters)
        
        toExport = pd.DataFrame(assignments)
        # TODO: make sure filepath matches directories in Wendian
        toExport.to_csv(f'Cluster Assignments - {str(R1)} to {str(R2)} - {edge} - {seg}.csv')
        
        with open(f'Cluster Sizes - {str(R1)} to {str(R2)} - {edge} - {seg}.csv', 'w') as f:
            f.write('Cluster,Count\n')
            f.writelines(f'{str(clusters[i])},{str(counts[i])}\n' for i in range(len(clusters)))
            f.close()
            
        clusterCounts = {}
        for i in range(len(clusters)):
            clusterCounts[clusters[i]] = counts[i]
            
        return assignments, clusterCounts
    
    def exportPR_RandomData(self, trial):
        assignments = pd.read_csv(f'{trial}/Random Cluster Assignments.csv')
        # check that this is correct
        
        clusterCounts = {}
        for i in assignments:
            clusterCounts[i] = 0
            for image in assignments[i]:
                if type(image) != float:
                    clusterCounts[i] += 1
            clusterCounts[i] = clusterCounts[i] / len(self.sampledImages)
        
        significantByType = {}
        weightedPsByType = {}
        weightedRsByType = {}
        for descriptorType in self.descriptorTypes:
            descriptors, splitByGenre = self.getClusterDescriptors(assignments, descriptorType)
            significant = self.getSpecificSignificantDescriptors(descriptors, descriptorType)
            
            truePos, falsePos = self.getTrueAndFalsePositives(descriptors, significant)
            falseNegs = self.getFalseNegatives(descriptors, significant)
            
            precisions, recalls = self.getPrecisionAndRecalls(truePos, falsePos, falseNegs)
            weightedP, weightedR = self.getWeightedPandR(precisions, recalls, clusterCounts)
            
            significantByType[descriptorType] = significant
            weightedPsByType[descriptorType] = [weightedP]
            weightedRsByType[descriptorType] = [weightedR]
            
        for descriptorType in significantByType:
            maxSize = max([len(i) for i in list(significantByType[descriptorType].values())])
            for cluster in significantByType[descriptorType]:
                while len(significantByType[descriptorType][cluster]) < maxSize:
                    significantByType[descriptorType][cluster].append(np.nan)
            clusterCols = pd.DataFrame(significantByType[descriptorType])
            clusterRows = clusterCols.transpose(copy=True)
            clusterRows.to_csv(f'{trial}/Significant Descriptors for {descriptorType} for Random Clusters.csv', header=False)
            
        Ps = pd.DataFrame(weightedPsByType)
        Ps = Ps.transpose()
        Ps.columns = ['Precision']
        Ps.to_csv('{trial}/Random Weighted Precisions By Type.csv')
        
        Rs = pd.DataFrame(weightedRsByType)
        Rs = Rs.transpose()
        Rs.columns = ['Recall']
        Rs.to_csv('{trial}/Random Weighted Recalls By Type.csv')
            
            
        
    
    #TODO: test this function
    def exportPRData(self, assignments, clusterCounts, edge, seg, R1, R2):
        # for i in clusterCounts:
        #     clusterCounts[i] = clusterCounts[i] / len(self.sampledImages)
        
        significantByType = {}
        weightedPsByType = {}
        weightedRsByType = {}
        for descriptorType in self.descriptorTypes:
            if descriptorType == 'Genre Specific':
                splitByGenre = True
            else:
                splitByGenre = False
                
            # get cluster descriptors of that type
            descriptors, splitByGenre = self.getClusterDescriptors(assignments, descriptorType, splitByGenre)
            # find the significiant descriptors per cluster; to export
            significant = self.getSpecificSignificantDescriptors(descriptors, descriptorType, splitByGenre)
            
            # get TPs, FPs, and FNs
            truePositives, falsePositives = self.getTrueAndFalsePositives(descriptors, significant, splitByGenre)
            falseNegatives = self.getFalseNegatives(descriptors, significant, splitByGenre)
            # get Ps and Rs
            precisions, recalls = self.getPrecisionAndRecalls(truePositives, falsePositives, falseNegatives, splitByGenre)
            weightedP, weightedR = self.getWeightedPandR(precisions, recalls, clusterCounts)
            
            if splitByGenre == False:
                significantByType[descriptorType] = significant
                weightedPsByType[descriptorType] = [weightedP]
                weightedRsByType[descriptorType] = [weightedR]
            else:
                for genre in weightedP:
                    genreName = f'Genre Specific - {genre}'
                    weightedPsByType[genreName] = weightedP[genre]
                    weightedRsByType[genreName] = weightedR[genre]
                    significantByType[genreName] = {}
                    for cluster in significant:
                        if genre in list(significant[cluster].keys()):
                            significantByType[genreName][cluster] = significant[cluster][genre]
                            
                    #TODO: get entropy for each genre
                            
        for descriptorType in significantByType:
            maxSize = max([len(i) for i in list(significantByType[descriptorType].values())])
            for cluster in significantByType[descriptorType]:
                while len(significantByType[descriptorType][cluster]) < maxSize:
                    significantByType[descriptorType][cluster].append(np.nan)
            clusterCols = pd.DataFrame(significantByType[descriptorType])
            clusterRows = clusterCols.transpose(copy=True)
            clusterRows.to_csv(f'Movements/Historical Movements/Significant Descriptors of {descriptorType} for Historical Movements.csv')
            # clusterRows.to_csv(f'Significant Descriptors for {descriptorType} - {str(R1)} to {str(R2)} - {edge} - {seg}.csv', header=False)
            
        Ps = pd.DataFrame(weightedPsByType)
        Ps = Ps.transpose()
        Ps.columns = ['Precision']
        Ps.to_csv('Movements/Historical Movements/Weighted Precisions By Type - Historical Movements.csv')
        
        Rs = pd.DataFrame(weightedRsByType)
        Rs = Rs.transpose()
        Rs.columns = ['Recall']
        Rs.to_csv('Movements/Historical Movements/Weighted Recalls By Type - Historical Movements.csv')
            
    
        
