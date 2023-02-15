# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:17:21 2022

@author: khloe
"""

import pandas as pd
import numpy as np
# from scipy.stats import skew

class PostClustering():
    '''
    Do as batch and parallelize over hyperparameters. os.chdir to trial.
    '''
    
    def __init__(self, sampleSize, sampledImages, random=False):
        self.sampleSize = sampleSize # --> int: 100, 300, or 500
        self.sampled = sampledImages # --> list of str
        self.descriptorTypes = ['All Formalist Descriptors', 'All Critical Descriptors',
                                'Abstraction Level', 'Genres', 'Descriptors', 'Color',
                                'Line Form and Value', 'Texture', 'Shape and Space']
        self.random = random
        self.dataType = 'Real'
        if self.random == True:
            self.dataType = 'Random'
        
    def __str__(self):
        return 'Post Clustering Class for Random and Real Data'
    
    def getClusterData(self, seg, edge, R1, R2):
        if self.random == False:
            assignments = pd.read_csv(f'Cluster Assignments - {str(R1)} to {str(R2)} - {edge} - {seg}.csv').drop(columns=['Unnamed: 0'])
            clusterCounts = pd.read_csv(f'Cluster Sizes - {str(R1)} to {str(R2)} - {edge} - {seg}.csv').set_index('Cluster')
        
            #change cluster counts to percentages of total - check that this works properly
            total = self.sampleSize
            counts = pd.DataFrame(columns=['Cluster', 'Count']).set_index('Cluster')
            for i in clusterCounts.index:
                counts.loc[i] = clusterCounts.loc[i] / total
                
        else:
            assignments = pd.read_csv(f'Random Cluster Assignments - {str(R1)} to {str(R2)} - {edge} - {seg}.csv').drop(columns=['Unnamed: 0'])
            total = self.sampleSize
            counts = pd.DataFrame(columns=['Cluster', 'Count']).set_index('Cluster')
            for cluster in assignments:
                # print(cluster)
                count = 0
                for image in assignments[cluster]:
                    if type(image) != float:
                        count += 1
                counts.loc[cluster] = count / total
        
        # print(assignments)    
            
        return assignments, counts
    
    def getClusterDescriptors(self, assignments, descriptorType):
        # filepath edits
        if descriptorType == 'All Formalist Descriptors' or descriptorType=='All Critical Descriptors':
            fp = ''
        else:
            fp = ' FINAL'
            
        allDescriptors = pd.read_csv(f'Movements/New Descriptors/Final Descriptors/{descriptorType}{fp}.csv', dtype=str).set_index('Name')
        sampleDescriptors = allDescriptors.loc[self.sampled]
        #print(sampleDescriptors.to_string())
        descriptors = {} #all descriptors in all the clusters: {cluster: {descriptor: count}}
        for cluster in assignments:
            descriptors[cluster] = {}
            for image in assignments[cluster]:
                # print(image, flush=True)
                if type(image) != float:
                    imageDescription = list(sampleDescriptors.loc[image])
                    # print(imageDescription, flush=True)
                    for word in imageDescription:
                        #if not nan
                        if type(word) != float:
                            if word not in list(descriptors[cluster].keys()):
                                descriptors[cluster][word] = 1
                            else:
                                descriptors[cluster][word] += 1
        return descriptors # should have no nan values -- check this
    
    def getSignificantDescriptors(self, descriptors, descriptorType, sigmaCutOff=1.5):
        significant = {}
        for cluster in descriptors:
            words = list(descriptors[cluster].keys())
            counts = list(descriptors[cluster].values())
            # print(words)
            if len(words) > 1:
                mean = np.mean(counts)
                std = np.std(counts)
                
                sigma = np.array([(i - mean) / std for i in counts])
                specific = np.where(sigma > sigmaCutOff)[0]
                specificWords = list(map(words.__getitem__, specific))
                # print(specificWords)
            elif len(words) == 1:
                specificWords = words
            else:
                specificWords = [np.nan]
                
            significant[cluster] = specificWords
                
        return significant
    
    def getTrueAndFalsePositives(self, descriptors, significant):
        truePos = {}
        falsePos = {}
        for cluster in descriptors:
            countTP = 0
            countFP = 0
            sigmas = significant[cluster]
            for word in descriptors[cluster]:
                if word in sigmas:
                    countTP += descriptors[cluster][word]
                else:
                    countFP += descriptors[cluster][word]
            truePos[cluster] = countTP
            falsePos[cluster] = countFP
            
        return truePos, falsePos
    
    def getFalseNegatives(self, descriptors, significant):
        falseNegs = {}
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
            falseNegs[cluster] = count
            
        return falseNegs
    
    def getPrecisionAndRecall(self, truePos, falsePos, falseNeg):
        precisions = {}
        recalls = {}
        for cluster in truePos:
            TP = truePos[cluster]
            if TP == 0:
                P = 0
                R = 0
            else:
                FP = falsePos[cluster]
                FN = falseNeg[cluster]
                
                P = TP / (TP + FP)
                R = TP / (TP + FN)
            
            precisions[cluster] = P
            recalls[cluster] = R
            
        return precisions, recalls
    
    def getAverages(self, precisions, recalls, clusterCounts):
        # print(precisions, clusterCounts)
        weightedP = 0
        weightedR = 0
        for cluster in precisions:
            # if self.random == False:
            #     num = int(cluster)
            # else:
            #     num = cluster
            # weightedP += clusterCounts['Count'].loc[clusterCounts['Cluster'] == cluster] * precisions[cluster]
            # weightedR += clusterCounts['Count'].loc[clusterCounts['Cluster'] == cluster] * recalls[cluster]
            weightedP += (49 / 734) * precisions[cluster]
            weightedR += (49 / 734) * recalls[cluster]
            
        avgP = np.mean(list(precisions.values()))
        avgR = np.mean(list(recalls.values()))
        return avgP, avgR, weightedP, weightedR
    
    def saveDictAsDF(self, dictionary, name, edge, seg, R1, R2):
        df = pd.DataFrame(dictionary)
        df = df.transpose()
        df.columns = [name]
        if edge == None:
            df.to_csv(f'Movements/New Descriptors/Historical Movements {name} By Type.csv')
        else:
            df.to_csv(f'{self.dataType} {name} By Type - {str(R1)} to {str(R2)} - {edge} - {seg}.csv')
    
    def exportPRData(self, edge, seg, R1, R2):
        assignments, clusterCounts = self.getClusterData(seg, edge, R1, R2)
        
        significantByType = {}
        avgPsByType = {}
        avgRsByType = {}
        weightedPsByType = {}
        weightedRsByType = {}
        for descriptorType in self.descriptorTypes:
            descriptors = self.getClusterDescriptors(assignments, descriptorType)
            significant = self.getSignificantDescriptors(descriptors, descriptorType)
            
            truePos, falsePos = self.getTrueAndFalsePositives(descriptors, significant)
            falseNegs = self.getFalseNegatives(descriptors, significant)
            
            precisions, recalls = self.getPrecisionAndRecall(truePos, falsePos, falseNegs)
            avgP, avgR, weightedP, weightedR = self.getAverages(precisions, recalls, clusterCounts)
            
            significantByType[descriptorType] = significant
            avgPsByType[descriptorType] = [avgP]
            avgRsByType[descriptorType] = [avgR]
            weightedPsByType[descriptorType] = [weightedP]
            weightedRsByType[descriptorType] = [weightedR]
        
        for descriptorType in significantByType:
            maxSize = max([len(i) for i in list(significantByType[descriptorType].values())])
            for cluster in significantByType[descriptorType]:
                while len(significantByType[descriptorType][cluster]) < maxSize:
                    significantByType[descriptorType][cluster].append(np.nan)
            clusterCols = pd.DataFrame(significantByType[descriptorType])
            clusterRows = clusterCols.transpose(copy=True)
            clusterRows.to_csv(f'Significant Descriptors for {descriptorType} for {self.dataType} Clusters - {str(R1)} to {str(R2)} - {edge} - {seg}.csv',
                               header=False) 
        
        self.saveDictAsDF(avgPsByType, 'Average Precisions', edge, seg, R1, R2)
        self.saveDictAsDF(avgRsByType, 'Average Recalls', edge, seg, R1, R2)
        self.saveDictAsDF(weightedPsByType, 'Weighted Precisions', edge, seg, R1, R2)
        self.saveDictAsDF(weightedRsByType, 'Weighted Recalls', edge, seg, R1, R2)
        
    def exportPRData_movements(self, assignments, clusterCounts):
        
        significantByType = {}
        avgPsByType = {}
        avgRsByType = {}
        weightedPsByType = {}
        weightedRsByType = {}
        for descriptorType in self.descriptorTypes:
            descriptors = self.getClusterDescriptors(assignments, descriptorType)
            significant = self.getSignificantDescriptors(descriptors, descriptorType)
            
            truePos, falsePos = self.getTrueAndFalsePositives(descriptors, significant)
            falseNegs = self.getFalseNegatives(descriptors, significant)
            
            precisions, recalls = self.getPrecisionAndRecall(truePos, falsePos, falseNegs)
            avgP, avgR, weightedP, weightedR = self.getAverages(precisions, recalls, clusterCounts)
            
            significantByType[descriptorType] = significant
            avgPsByType[descriptorType] = [avgP]
            avgRsByType[descriptorType] = [avgR]
            weightedPsByType[descriptorType] = [weightedP]
            weightedRsByType[descriptorType] = [weightedR]
        
        for descriptorType in significantByType:
            maxSize = max([len(i) for i in list(significantByType[descriptorType].values())])
            for cluster in significantByType[descriptorType]:
                while len(significantByType[descriptorType][cluster]) < maxSize:
                    significantByType[descriptorType][cluster].append(np.nan)
            clusterCols = pd.DataFrame(significantByType[descriptorType])
            clusterRows = clusterCols.transpose(copy=True)
            clusterRows.to_csv(f'Movements/New Descriptors/Significant Descriptors {descriptorType} for Historical Movements.csv',
                               header=False) 
        
        self.saveDictAsDF(avgPsByType, 'Average Precisions', None, None, None, None)
        self.saveDictAsDF(avgRsByType, 'Average Recalls', None, None, None, None)
        self.saveDictAsDF(weightedPsByType, 'Weighted Precisions', None, None, None, None)
        self.saveDictAsDF(weightedRsByType, 'Weighted Recalls', None, None, None, None)
        
        
            
            
        
        
        
        
        
        
        
        
        
        
            
        
    
    
