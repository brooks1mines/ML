# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:31:27 2021

@author: khloe
"""
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import skimage.segmentation as seg
import pandas as pd
from skimage.measure import regionprops
import scipy
import networkx as nx
import glob
from skimage import io
import time

#----------------------  Image pre-processing class --------------------------      
#  - deals with single image matrix as input

class PreProcessing():
    '''
    This is the pre-processing class. It takes a single image matrix when 
    initialized. In this class, you can check the size of the image, and if 
    necessary, rescale the image to regularized value, IMAGE_HEIGHT * IMAGE_WIDTH.
    '''
    
    IMAGE_HEIGHT = 500
    IMAGE_WIDTH = 500
    
    def __init__(self, imMatrix):
        
        self.matrix = imMatrix
        self.imHeight = len(imMatrix)
        self.imWidth = len(imMatrix[0])
        
    def __str__(self):
        return "Pre-processing Class"
    
    def checkSize(self, height = IMAGE_HEIGHT, width = IMAGE_WIDTH):
        
        '''

        Parameters
        ----------
        height : int, optional
            Pixel height that we are checking the image against. 
            The default is IMAGE_HEIGHT = 500.
        width : int, optional
            Pixel width we check the image against. 
            The default is IMAGE_WIDTH = 500.

        Returns
        -------
        correctSize : bool
            if true, image is correct size; if false, image needs 
            to be resized.

        '''
        
        if (self.imHeight == height) and (self.imWidth == width):
            correctSize = True
        else:
            correctSize = False
        
        return correctSize
    
    #future TODO: set acceptable change in entropy?
    def rescaleImage(self, height = IMAGE_HEIGHT, width = IMAGE_WIDTH, interpType = 'PAR'):
        '''

        Parameters
        ----------
        height : int, optional
            Desired pixel height of image. The default is IMAGE_HEIGHT.
        width : int, optional
            Desired pixel width of image. The default is IMAGE_WIDTH.
        interpType : str, optional
            Interpolation method to use. Options:
                1. 'NN' - nearest neighbor
                2. 'BL' - bilinear
                3. 'BC' - bicubic
                4. 'PAR' - pixel area relation 
            The default is 'PAR'.

        Raises
        ------
        Exception
            If interpolation type entered is not an option.

        Returns
        -------
        newMat : 3-layer matrix of ints
            DESCRIPTION.

        '''
        
        rescale = self.checkSize()
        
        if rescale == False:
            
            #get shannon entropy of the orginial matrix
            # ent1 = entropy(self.matrix)
            # pV1 = ent1.getPixelVectors()
            # S1 = ent1.first_order(ent1.getPDF(pV1[0]))
            
            #change size of image to desired size
            dim = (height, width)
            
            if interpType == 'NN':
                newMat = cv2.resize(self.matrix, dim, interpolation = cv2.INTER_NEAREST)
                
            elif interpType == 'BL':
                newMat = cv2.resize(self.matrix, dim, interpolation = cv2.INTER_LINEAR)
                
            elif interpType == 'BC':
                newMat = cv2.resize(self.matrix, dim, interpolation = cv2.INTER_CUBIC)
                
            elif interpType == 'PAR':
                newMat = cv2.resize(self.matrix, dim, interpolation = cv2.INTER_AREA)
                
            else:
                raise Exception("Incorrect interpolation method choice. Can choose from the following:\n NN\n BL\n BC\n PAR")
                
            #make sure matrix elements are 0 <= x <= 255
            for i in range(height):
                for j in range(width):
                    for k in range(3):
                        
                        #some interpolation methods will return floats rather than ints
                        newMat[i][j][k] = int(newMat[i][j][k])
                        
                        if newMat[i][j][k] < 0:
                            newMat[i][j][k] = 0
                                
                        elif newMat[i][j][k] > 255:
                            newMat[i][j][k] = 255
            
            # assign new matrix as object
            self.matrix = newMat
            
            return newMat 
            
        else:
            
            print("Image is already of the desired pixel size.")
            return self.matrix
                

#----------------------  Network Creation class ------------------------------
#  - deals with filepath as object
class BuildNetwork():
    '''
    This class builds the representative complex network of the image. It takes
    the image matrix, image name and image movement as arguements upon 
    implementation. One can set the radial distance threshold by changing 
    self.R in the init function.
    
    Possible actions within this class:
        Image segmentation
        Show centers of segments in image
        Create network from one of 5 ECTs - radial distance, pixel size, color
            orientation, or entropy
        Identify color from HSL color vector - for color ECT
    '''
    
    #preprocessing should be done on image before calling this class
    def __init__(self, imMatrix, imName, imMovement, R1, R2):
        '''
        
        Parameters
        ----------
        imMatrix : 3-layer matrix of ints
            RGB matrices of the image, post-pre processessing
        imName : str
            Name of the image/painting
        imMovement : str
            Name of the movement that painting is from
        '''
        
        self.matrix = imMatrix
        self.name = imName
        self.movement = imMovement
        self.R1 = R1
        self.R2 = R2
        
        
    #if class variable printed, return movement: name
    def __str__(self):
        return self.movement + ': ' + self.name
    
    # perform image segmentation of given type
    def imageSegmentation(self, segType):
        '''

        Parameters
        ----------
        segType : str
            Choice of image segmentation method. Choose one of the following:
                1. 'Felz' - Felzenswalb
                2. 'Quick' - Quickshift
                3. 'SLIC' - SLIC
                4. 'Wat' - Watershed

        Raises
        ------
        Exception
            If inappropriate str input for segType.

        Returns
        -------
        image : 3-layer matrix of ints
            image post-segmentaiton. Will have 'segments' embedded in the image.
            
        Notes
        -----
        Image must be a matrix of floats (H, W, 3); techniques dont work on int images.

        '''
        
          
        #Felzenswalb
        if segType == 'Felz':
            image = seg.felzenszwalb(self.matrix, scale = 300, min_size = 200)
        
        #Quickshift
        # sigma --> width for gaussian smoothing as preprocessing
        # max-dist --> cut off point for data distances, higher = fewer clusters
        # ratio --> balances color space and image space proximity, higher = more weight to color space
        elif segType == 'Quick':
            image = seg.quickshift(self.matrix, kernel_size=6, max_dist=500, ratio=0.5)
        #SLIC
        elif segType == 'SLIC':
            image = seg.slic(self.matrix, sigma=1, start_label=(0), compactness = 10)
           
        else:
            raise Exception("Incorrect segmentation choice.")
                
        return image
    
    #show outline of the superpixels and their centers
    def showCenterOfSegments(self, superPixelMatrix):
        
        '''
        Parameters
        ---------
        superPixelMatrix : 3-layer matrix of ints
            matrix from segmentation techniques
            
        Returns
        -------
        regions : dict
            contains various characteristics of the superpixels
        centersX : list of ints
            x (column) value of center of the individual superpixels
        centersY : list of ints
            y (row) value of the center of the individual superpixels 
        '''
        
        regions = regionprops(superPixelMatrix, intensity_image=(self.matrix))
        
        centersX = []
        centersY = []
        
        for props in regions:
            
            cy = props.centroid[0] #rows
            cx = props.centroid[1] #cols
            
            centersY.append(cy)
            centersX.append(cx)
            
        plt.figure(figsize=(10, 10))
        plt.imshow(seg.mark_boundaries(self.matrix, superPixelMatrix))
        plt.plot(centersX, centersY, 'o', color = 'red')
        plt.show()
        
        return regions, centersX, centersY
    
    def _fast_edges(self, pos, radius, p = 2):
        '''
        from NetworkX github, modified by Khloe

        Parameters
        ----------
        pos : dict
            center positions (x, y) of each superpixel.
        radius : int
            This value defines the maximum radial (pixel) distance 
            to create an edge between two superpixels on the network.
        p : int, optional
            The space we funciton in. The default is 2, for Euclidean space.

        Returns
        -------
        edges : list of tuples of ints
            Describes the nodes to have edges between them.

        '''
        nodes, coords = list(zip(*pos.items()))
        kdtree = scipy.spatial.KDTree(coords)
        edge_indexes = kdtree.query_pairs(radius, p)
        edges = ((nodes[u], nodes[v]) for u, v in edge_indexes)
        return edges
    

    
    # pixel size network
    def makeNetwork_pixelSize(self, superPixelMatrix):
        '''

        Parameters
        ----------
        superPixelMatrix : matrix of ints
            Describes how the image is split into superpixels.

        Returns
        -------
        G : graph structure with node and edge objects
        
        NOTES
        -----
        This graph creates edges based on the amount of pixels within the superpixels. 
        These values are binned and those bin edges determine how the edges are created; 
        All nodes within the same bin are connected. An added caviat is that the centers
        of the nodes within the same bin need to be within a certain pixel radius from 
        each other, chosen as 100 pixels for now.

        '''
        
        regions, centX, centY = self.showCenterOfSegments(superPixelMatrix)
        
        #labels for the nodes --> correlated to the superpixels
        labels = [r.label for r in regions]
        areas = [r.area for r in regions]
        
        #create the network... only doing absolute binning since images will all have the same size
        G = nx.Graph()
        
        pos = {}
        areasDict = {}
        
        for i in labels:
            G.add_node(i)
            pos[i] = (centX[i - 1], -centY[i - 1])
            areasDict[i] = areas[i - 1]
          
        if len(areas) != 0:
            # find how the pixel sizes should be binned --> how edges should be created
            bin_amount = int(round(np.sqrt(len(areas))))
            hist, bin_edges = np.histogram(areas, bins = bin_amount)
            
            # plt.hist(areas, bins = bin_amount)
            # plt.show()
        
        edges = []
            
        # now, for each node, compare to the other nodes
        # check if the two nodes are within the same bin
        #for each bin from the histogram
        for i in range(len(bin_edges)-1):
            #find upper and lower edges of that bin
            low_lim = int(np.round(bin_edges[i]))
            upper_lim = int(np.round(bin_edges[i+1]))
            
            ps_range = range(low_lim, upper_lim + 1)
            for key in areasDict:
                for okey in areasDict:
                    # dont want self loops
                    if key != okey:
                        # dont want to repeat an edge, since this is a single layer graph
                        if ((key, okey) not in edges) and ((okey, key) not in edges):
                        
                            radial_dist = np.sqrt(math.pow(centX[key - 1] - centX[okey - 1], 2) + math.pow(centY[key - 1] - centY[okey - 1], 2))
                            # if the nodes are within a certain range of radial distance from eachother
                            if radial_dist <= self.R2 and radial_dist >= self.R1:
                                
                                # if both within same bin
                                if areasDict[key] in ps_range and areasDict[okey] in ps_range:
                                    edges.append((key, okey))
                                    
                                
        G.add_edges_from(edges)
        # nx.set_edge_attributes(G, attr)
        # edges1,weights = zip(*nx.get_edge_attributes(G,'weight').items())
        
        nx.draw(G, pos, nodelist = labels, with_labels = True) #, edge_color=weights, edge_cmap=plt.cm.inferno)
        plt.show()
            
        return G
    
    
    def RGBtoHSL(self, pixelVector):
        '''
        Parameters
        ----------
        pixelVector : 3 tuple of ints
            RGB color description of a single pixel.

        Returns
        -------
        H : int
            hue, 0 < H <= 360.
        S : float
            saturation, 0 <= S <= 1.
        L : float
            lightness, 0 <= L <= 1.

        '''
        
        R, G, B = pixelVector[0] / 255, pixelVector[1] / 255, pixelVector[2] / 255
        
        pixelVals = [R, G, B]
        
        M = max(pixelVals)
        m = min(pixelVals)
        
        C = M - m
        
        if (R == G) and (G == B):
            Hprime = 0
        
        elif M == R:
            Hprime = ((G - B) / C) % 6
            
        elif M == G:
            Hprime = ((B - R)/ C) + 2
            
        elif M == B:
            Hprime = ((R - G)/ C) + 4

        else:
            
            Hprime = 0
            
        H = round(60 * Hprime)
        
        L = ((M + m) / 2)
        
        if L in (0, 1):
            S = 0
        else:
            S = round((C / (1 - np.abs((2 * L) - 1))) * 100)
            
        L = round(L * 100)
            
        return (H, S, L)
    
    # identify the color of a pixel vector in RBG
    def identifyColor(self, pixelVector, groups = 'Color Wheel'):
        '''

        Parameters
        ----------
        groups : str, optional
            What color aspect you'd like to identify. The default is 'Color Wheel'.
            Other options are Lightness or Saturation
            
        pixelVector : list of ints (0 to 255)
            vector that describes the RGB values of a pixel

        Returns
        -------
        None.
        

        '''
        
        # convert RBG to HSL
        HSL = self.RGBtoHSL(pixelVector)
        
        groupsVals = pd.read_csv(f"Entropy/Understanding Entropy/Diversity/Color RGB Groups/{groups} Key.csv", header=None)
        groupsVals = groupsVals.set_index(0)
         
        # what is the user looking to identify ?     
        if groups == 'Saturation':
            c = 1
        elif groups == 'Lightness':
            c = 2
        else:
            c = 0
            
        # assign identity
        for i in groupsVals.index:
            
            # light and sat vals should exclusively go to this if statement rather than the else
            if groupsVals.at[i, 2] > groupsVals.at[i, 1]:
                
                maxVal = groupsVals.at[i, 2]
                minVal = groupsVals.at[i, 1]
                
                if HSL[c] >= minVal and HSL[c] < maxVal:
                    
                    identity = i
                    break
                
            # should only see hue because 'red' wraps around H vals 
            else:
                minVal_1 = groupsVals.at[i, 1]
                maxVal_1 = 361
                
                minVal_2 = 0
                maxVal_2 = groupsVals.at[i, 2]
                
                if (HSL[c] >= minVal_1 and HSL[c] < maxVal_1) or (HSL[c] >= minVal_2 and HSL[c] < maxVal_2):
                    
                    identity = i
                    break
              
                
        return identity
                
    
    # color palette network -- theres probably a better way to do this
    def makeNetwork_color(self, superPixelMatrix):
        '''

        Parameters
        ----------
        superPixelMatrix : matrix of ints
            Describes how the image is split into superpixels.

        Returns
        -------
        G : graph structure with node and edge objects
        
        NOTES
        -----
        This graph creates edges based on the average color of the superpixel. (The
        average is taken from the R, G, B individually and combined in a vector.) We find 
        the color and determine which color palette(s) that superpixel then belongs to. If
        any two nodes are within a certain pixel radius and from the same color palette, 
        they have an edge.

        '''
        
        #get superpixel information
        regions, centX, centY = self.showCenterOfSegments(superPixelMatrix)
        
        labels = [r.label for r in regions]
        areas = [r.area for r in regions]
        avgs = np.array([r.mean_intensity for r in regions])
                       
        # identify the average color of each superpixel
        superPixels = {}
        for i in range(len(labels)):
            
            hue = self.identifyColor(avgs[i])
            sat = self.identifyColor(avgs[i], 'Saturation')
            light = self.identifyColor(avgs[i], 'Lightness')
            
            superPixels[labels[i]] = [hue, sat, light]
        
            
        colorDict = pd.read_csv('Entropy/Understanding Entropy/Diversity/Color RGB Groups/Color Dictionary.csv', header=None)
        colorDict = colorDict.set_index(3)
        
        # index each superpixel based on its color
        pixelIndexing = {}
        for i in superPixels:
            if superPixels[i][2] == 'white':                 
                pixelIndexing[i] = 111
                
            elif superPixels[i][2] == 'black':
                pixelIndexing[i] = 110
                
            elif superPixels[i][1] == 'grey':
                pixelIndexing[i] = 109
            
            else:
                for j in colorDict.index:
                    if np.array_equal(superPixels[i], np.array(colorDict.loc[j])):
                        pixelIndexing[i] = j
                        break
        
        # create complex network 
        G = nx.Graph()
        
        # # assign nodes and thier positions from superpixels and their centers
        pos = {}
        nodeSize = []
        for i in range(len(labels)):
            G.add_node(labels[i])
            pos[labels[i]] = (centX[i], -centY[i])
            nodeSize.append(areas[i])
        
        edges = []
        for i in pixelIndexing:
            for j in pixelIndexing:
                # no self loops
                if i != j:
                    #spatial embedding
                    radial_dist = np.sqrt(math.pow(np.abs(pos[i][0] - pos[j][0]),2) + math.pow(np.abs(pos[i][1] - pos[j][1]),2))
                    if radial_dist >= self.R1 and radial_dist <= self.R2:
                        # if within same color bin (i.e. have same color index in entry)
                        if pixelIndexing[i] == pixelIndexing[j]:
                            # check that edge doesn't already exist in edges list
                            if ((i, j) not in edges) and ((j, i) not in edges):
                                edges.append((i, j))
                        
        
        # # create edges on the complex network                  
        G.add_edges_from(edges)
        
        # # draw the complex network
        nx.draw(G, pos, nodelist = labels, with_labels = True)
        plt.show()
        
        # # return complex network object
        return G
    

    
    
    # superpixel orientation network
    def makeNetwork_orientation(self, superPixelMatrix):
        '''

        Parameters
        ----------
        superPixelMatrix : matrix of ints
            Describes how the image is split into superpixels.

        Returns
        -------
        G : graph structure with node and edge objects
        
        NOTES
        -----
        This graph creates edges based on the 'orientation' of the superpixels. Using
        algorithms available in the regionprops function (skiimage.measure package), the code
        estimates the superpixel as an ellipse and finds the angle of that shape's major axis 
        from the 'x'-axis. Then, if any two nodes are of the same (region of) angle and within
        some radius, they are connected.

        '''
        
        # get superpixel information
        regions, centX, centY = self.showCenterOfSegments(superPixelMatrix)
        
        labels = [r.label for r in regions]
        areas = [r.area for r in regions]
        angle = [r.orientation for r in regions]
        
        # create dictionary that tells what angle each superpixel's equivalent ellipse has
        angleDict = {}
        for i in labels:
            angleDict[i] = angle[i - 1]
            
        # create complex network    
        G = nx.Graph()
        
        # assign nodes and positions from superpixels
        pos = {}
        nodeSize = []
        for i in range(len(labels)):
            G.add_node(labels[i])
            pos[labels[i]] = (centX[i], -centY[i])
            nodeSize.append(areas[i])
        
        # binning metrics for edge creation
        angles = [((-np.pi / 2) + (np.pi / 24), -np.pi / 4), (-np.pi / 4, -np.pi / 24),
                   (np.pi / 24, np.pi / 4), (np.pi / 4, (np.pi / 2) - (np.pi / 24))]
        
        edges = []
        for i in angles:
            for j in angleDict:
                for k in angleDict:
                    if j != k and ((j, k) not in edges and (k, j) not in edges):
                        
                        radial_dist = np.sqrt(math.pow(np.abs(pos[j][0] - pos[k][0]),2) + math.pow(np.abs(pos[j][1] - pos[k][1]),2))
                        
                        if (radial_dist >= self.R1) and (radial_dist <= self.R2):
                            if (angleDict[j] >= i[0] and angleDict[j] < i[1]) and (angleDict[k] >= i[0] and angleDict[k] < i[1]):
                                edges.append((j, k))
                            
        # print(edges)
        G.add_edges_from(edges)   
        nx.draw(G, pos, nodelist = labels, with_labels = True)
        
        plt.show()
        
        return G
    
    

#----------------------  Network Measurements class -------------------------- 
# - deals with complex network as object
class Measurements():
    '''
    This class contains all measurement functions for the complex networks. It
    takes a networkX graph object and the ECT used upon implementation. 
    implementation. There is a function to call that will take the user 
    desired measurement.
    '''
    
    
    def __init__(self, G, edge_type):
        '''
        Parameters
        ----------
        G : graph structure
            Complex network made from the buildNetwork class.
            
        edge_type : str
            Describes the type of edges
            
        Returns
        -------
        none.
        '''
        
        self.network = G
        self.is_directed = nx.is_directed(G)
        self.edge_type = edge_type
        
    def __str__(self):
        
        return self.edge_type
    
    # degree distribution
    def degree_dist(self):
        '''

        Returns
        -------
        mean : float
            Normalized average value of the network's degree distribution.
        variance : float
            Normalized variance value of the network's degree distribution.
        skew : float
            Normalized skewness value of the network's degree distribution.
        kurtosis : float
            Normalized kurtosis value of the network's degree distribution.

        '''
        
        degrees = [self.network.degree(n) for n in self.network.nodes()]
        
        mean = np.mean(degrees) / len(degrees)
        variance = scipy.stats.moment(degrees, moment = 2)
        skew = scipy.stats.moment(degrees, moment = 3)
        kurtosis = scipy.stats.moment(degrees, moment = 4)
        
        return mean, variance, skew, kurtosis
        
    # clustering 
    def clustering_dist(self):
        '''
        Returns
        -------
        mean : float
            Normalized average value of the network's clustering distribution.
        variance : float
            Normalized variance value of the network's clustering distribution.
        skew : float
            Normalized skewness value of the network's clustering distribution.
        kurtosis : float
            Normalized kurtosis value of the network's clustering distribution.

        '''
        
        #should I do a weighted average for the clustering coeff or leave as is?
        
        clusters = [nx.clustering(self.network, n) for n in self.network.nodes()]
        
        mean = np.mean(clusters) / len(clusters)
        variance = scipy.stats.moment(clusters, moment = 2)
        skew = scipy.stats.moment(clusters, moment = 3)
        kurtosis = scipy.stats.moment(clusters, moment = 4)
        
        return mean, variance, skew, kurtosis
    
    # cliques
    def get_clique_info(self):
        '''

        Returns
        -------
        cliqueNumber : float
            This is the size (number of nodes involved) of the largest clique in the network.
            To be a clique, the subgraph (where all nodes are adjacent to the others) must have 
            at least 3 nodes. Normalized by the number of nodes in the graph.
        numCliques : int
            This is the number of cliques (3 nodes or greater) in the image.

        '''
        
        cliques = list(nx.find_cliques(self.network))
        
        #list of cliques that contain 3 or more nodes
        actualCliques = []
            
        for i in range(len(cliques)):
            if len(cliques[i]) > 2:
                actualCliques.append(cliques[i])
            
        numCliques = 0        
            
        #make sure there actually are cliques in the network, otherwise, error
        # # if list isnt empty
        if actualCliques:
            
            cliqueNumber = len(actualCliques[0])
        
            for i in actualCliques:
                numCliques += 1
                if len(i) > cliqueNumber:
                    cliqueNumber = len(i)
                    
            # normalizing
            cliqueNumber = cliqueNumber / self.get_number_nodes()
        
        #if list is empty 
        else:
            cliqueNumber = 0
                
        return cliqueNumber, numCliques
        

    # largest component
    def get_component_info(self):
        '''

        Returns
        -------
        numComp : int
            This is the number of components in the network.
        largest : float
            This is the size (number of nodes) of the largest component.
            Normalized by the number of nodes in the total network.

        '''
        
        components = list(nx.connected_components(self.network))
        
        numComp = len(components)
        
        if numComp > 0:
            largest = len(components[0])
            for i in components:
                if len(i) > largest:
                    largest = len(i)
                    
            #normalizing
            largest = largest / self.get_number_nodes()
            
        else:
            largest = 0
                
        return numComp, largest
        

    # assortativity
    def get_assortativity(self):
        '''

        Returns
        -------
        degreeAssort : float
            The tendency for nodes to be connected to those of the same level (high or low) of degree.
            Possible values from -1 to 1.

        '''
        
        if len(self.network.edges()) != 0:        
        
            degreeAssort = nx.degree_assortativity_coefficient(self.network)
            
        else:
            degreeAssort = 0
        
        return degreeAssort

    # centralities
    def get_centralities(self):
        '''

        Returns
        -------
        degree : dict
            This returns a distribution of the degree centrality for each node.
        close : dict
            This returns a distribution of the closeness centrality for each node.
        between : dict
            This returns a distribution of the betweenness centrality for each node.

        '''
        degree = nx.degree_centrality(self.network)
        close = nx.closeness_centrality(self.network)
        between = nx.betweenness_centrality(self.network)
        
        return degree, close, between
    
    # number of nodes 
    def get_number_nodes(self):
        '''

        Returns
        -------
        int
            Total number of nodes in the graph.

        '''
        return len(list(self.network.nodes()))
    
    # bridges
    def get_number_bridges(self):
        '''
        Returns
        -------
        float
            Total number of bridges in the graph normalized by the number of nodes.
        '''
        
        numBridges = 0
        
        if nx.has_bridges(self.network):
            numBridges = len(list(nx.bridges(self.network))) / self.get_number_nodes()
        
        return numBridges
        
    # density
    def get_density(self):
        '''

        Returns
        -------
        float
            The edge density of the graph. In other words, this return the fraction
            of edges that exist over the number of all possible edges.

        '''
        return nx.density(self.network) 
    
    # function to determine critical point? Think about the dispersion map used to 
    # identify quantum phase transitions plus think about complexity book
    
    def takeChosenMeasurements(self, types):
        '''

        Parameters
        ----------
        types : list of str
            This describes the measurements to be done on the graph. This should 
            be a two dimensional list.

        Raises
        ------
        Exception
            For invalid measurement choice. Check types parameter for misspellings.

        Returns
        -------
        Measurements : dict
            This dict contains all measurements requested.

        '''
        # function that will only perform the chosen measurements and return those
        # instead of performing all and only returning a few - will hopefully 
        # lessen the runtime
        
        Measurements = {}
        
        for i in types:
            
            # degree - average, variance, skew, kurtosis
            if i[0] == "Degree":
                mean, variance, skew, kurtosis = self.degree_dist()
                
                if i[1] == "Average":
                    Measurements['Degree Average'] = mean
                    
                elif i[1] == "Variance":
                    Measurements['Degree Variance'] = variance
                    
                elif i[1] == "Skew":
                    Measurements['Degree Skew'] = skew
                    
                elif i[1] == "Kurtosis":
                    Measurements['Degree Kurtosis'] = kurtosis
                    
                else:
                    raise Exception("Invalid degree statstic chosen. Options are: 'Average', 'Variance', 'Skew', or 'Kurtosis'.")
                
            # clustering - average, variance, skew, kurtosis
            elif i[0] == "Clustering":
                mean, variance, skew, kurtosis = self.clustering_dist()
                
                if i[1] == "Average":
                    Measurements['Clustering Average'] = mean
                    
                elif i[1] == "Variance":
                    Measurements['Clustering Variance'] = variance
                    
                elif i[1] == "Skew":
                    Measurements['Clustering Skew'] = skew
                    
                elif i[1] == "Kurtosis":
                    Measurements['Clustering Kurtosis'] = kurtosis
                    
                else:
                    raise Exception("Invalid clustering statstic chosen. Options are: 'Average', 'Variance', 'Skew', or 'Kurtosis'.")
            
            # cliques - clique number, number of cliques
            elif i[0] == "Cliques":
                
                cliqueNum, numCliques = self.get_clique_info()
                
                if i[1] == "Clique Number":
                    Measurements['Clique Number'] = cliqueNum
                    
                elif i[1] == "Number of Cliques":
                    Measurements['Number of Cliques'] = numCliques
                    
                else:
                    raise Exception("Invalid clique statistic chosen. Options are: 'Clique Number' or 'Number of Cliques'.")
            
            # components - number of components, largest
            elif i[0] == "Components":
                
                numComp, largest = self.get_component_info()
                
                if i[1] == "Number of Components":
                    Measurements['Number of Components'] = numComp
                    
                elif i[1] == "Largest":
                    Measurements['Largest Component Size'] = largest
                    
                else:
                    raise Exception("Invalid component statistic chosen. Options are: 'Number of Components' or 'Largest'.")
            
            # degree assortativity
            elif i[0] == "Assortativity":
                
                Measurements['Degree Assortativity'] = self.get_assortativity()
            
            # centralities - degree, closeness, betweeness
            elif i[0] == 'Centrality':
                
                degree, close, between = self.get_centralities()
                
                if i[1] == 'Degree':
                    
                    values = list(degree.values())
                    
                    if i[2] == 'Average':
                        mean = np.mean(values) / len(values)
                        Measurements['Degree Centrality Average'] = mean
                    
                    elif i[2] == 'Variance':
                        variance = scipy.stats.moment(values, moment = 2)
                        Measurements['Degree Centrality Variance'] = variance
                        
                    elif i[2] == 'Skew':
                        skew = scipy.stats.moment(values, moment = 3)
                        Measurements['Degree Centrality Skew'] = skew
                        
                    elif i[2] == 'Kurtosis':
                        kurtosis = scipy.stats.moment(values, moment = 4)
                        Measurements['Degree Centrality Kurtosis'] = kurtosis
                    
                    else:
                        raise Exception("Invalid degree centrality statistic. Options are: 'Average', 'Variance', 'Skew', or 'Kurtosis'.")
                
                elif i[1] == "Closeness":
                    
                    values = list(close.values())
                    
                    if i[2] == 'Average':
                        mean = np.mean(values) / len(values)
                        Measurements['Closeness Centrality Average'] = mean
                    
                    elif i[2] == 'Variance':
                        variance = scipy.stats.moment(values, moment = 2)
                        Measurements['Closeness Centrality Variance'] = variance
                        
                    elif i[2] == 'Skew':
                        skew = scipy.stats.moment(values, moment = 3)
                        Measurements['Closeness Centrality Skew'] = skew
                        
                    elif i[2] == 'Kurtosis':
                        kurtosis = scipy.stats.moment(values, moment = 4)
                        Measurements['Closeness Centrality Kurtosis'] = kurtosis
                    
                    else:
                        raise Exception("Invalid closeness centrality statistic. Options are: 'Average', 'Variance', 'Skew', or 'Kurtosis'.")
                
                    
                elif i[1] == "Betweenness":
                    
                    values = list(between.values())
                    
                    if i[2] == 'Average':
                        mean = np.mean(values) / len(values)
                        Measurements['Betweenness Centrality Average'] = mean
                    
                    elif i[2] == 'Variance':
                        variance = scipy.stats.moment(values, moment = 2)
                        Measurements['Betweenness Centrality Variance'] = variance
                        
                    elif i[2] == 'Skew':
                        skew = scipy.stats.moment(values, moment = 3)
                        Measurements['Betweenness Centrality Skew'] = skew
                        
                    elif i[2] == 'Kurtosis':
                        kurtosis = scipy.stats.moment(values, moment = 4)
                        Measurements['Betweenness Centrality Kurtosis'] = kurtosis
                    
                    else:
                        raise Exception("Invalid betweenness centrality statistic. Options are: 'Average', 'Variance', 'Skew', or 'Kurtosis'.")
                
                    
                else:
                    raise Exception("Invalid centrality type. Options are: 'Degree', 'Closeness', or 'Betweenness'.")
                    
            
            # number of - nodes, bridges
            elif i[0] == "Number":
                
                if i[1] == "Nodes":
                    Measurements['Number of Nodes'] = self.get_number_nodes()
                    
                elif i[1] == 'Bridges':
                    Measurements['Number of Bridges'] = self.get_number_bridges()
                    
                else:
                    raise Exception("Invalid count type. Options are: 'Nodes' or 'Bridges'.")
            
            # density
            elif i[0] == 'Density':
                
                Measurements['Density of Edges'] = self.get_density()
                
            else:
                raise Exception("Invalid measurement type. Options are: 'Degree', 'Clustering', 'Cliques', 'Components', 'Assortativity', 'Centrality', 'Number', or 'Density'.")
            
        return Measurements
         
            
#---------------------- Data Handling Class ----------------------------------
# movements analysis
class TakeMeasurements():
    '''
    This class is what the user will call to build all networks, take all
    measurements, and export raw data. It takes a list of str (movements, as
    described in folder name) upon implementation. There is a function to build
    networks of the images within those movement folders, take measurements of
    those networks, and export the raw data to a csv, which is the only function
    the user should need to call.
    '''
    
    def __init__(self, movements, movements_dict=None):
        '''

        Parameters
        ----------
        movements : list of str
            Describes the movements to be analyzed. Flat list

        Returns
        -------
        None.

        '''
        
        self.meas_list = [['Degree', 'Average'], ['Degree', 'Variance'], ['Degree', 'Skew'],['Degree', 'Kurtosis'], 
                          ['Clustering', 'Average'], ['Clustering', 'Variance'], ['Clustering', 'Skew'], ['Clustering', 'Kurtosis'],
                          ['Cliques', 'Clique Number'], ['Cliques', 'Number of Cliques'], ['Components', 'Number of Components'],
                          ['Components', 'Largest'], ['Assortativity'], ['Centrality', 'Degree', 'Average'], ['Centrality', 'Closeness', 'Average'],
                          ['Centrality', 'Betweenness', 'Average'], ['Number', 'Nodes'], ['Number', 'Bridges'], ['Density']]
        
        # none if NOT doing bootstrapping
        if movements_dict is None:
        
            allImages = {}
            for i in movements:
                
                allImages[i] = []
                
                #need all image types
                for filename in glob.glob('Movements/Historical Movements/' + i + '/*.jpg'):
                    allImages[i].append(filename)
                    
                for filename in glob.glob('Movements/Historical Movements/' + i + '/*.jpeg'):
                    allImages[i].append(filename)
                    
                    
            self.numMovements = len(movements)
            self.movements = movements
            self.imagesDict = allImages
            self.images = []
            self.bootstrap = False
        
        else:
            
            unique_movements = movements['Movement'].unique()
            self.numMovements = len(unique_movements)
            self.movements = unique_movements
            self.imagesDict = movements_dict
            self.images = []
            self.bootstrap = True
        
        
        
    def __str__(self):
        
        print('Movements: ') 
        for i in self.movements:
            print(i + '\n')
            
        return "Dimensionality and Clustering Class"
            
    
    def flatten_list(self):
        '''

        Notes
        -----
        Creates a flat list of all images from all movements to be analyzed.

        '''
        allFlat = []
        
        for i in self.imagesDict:
            for j in self.imagesDict[i]:
                allFlat.append(j)
                
        self.images = allFlat
        
    def getImageFilepath(self):
        
        self.flatten_list()
        
        return self.images
    
    def getImageAndMovementNames(self):
        
        self.flatten_list()
        
        movements = []
        names = []
        
        for i in self.images:
            new = i.split('/')
            neww = new[2].split('\\')
            movement = neww[0]
            name = neww[1].split('.')[0]
            
            movements.append(movement)
            names.append(name)
            
        return movements, names
            
    
    def networkAndMeasurement(self, edge_type, seg_type, meas_list, R1, R2):
        '''

        Parameters
        ----------
        edge_type : str
            Users choice for edge creation technique.
            Options are: 'Pixel Size', 'Color', 'Orientation', 'Entropy'
        seg_type : str
            Users choice for image segmentation technique.
            Options are: 'Felz', 'Quick', 'SLIC'
        meas_list : list of str
            Two dimensional list that describes the measurements to be taken.
        Raises
        ------
        Exception
            If invalid edge creation technique chosen.

        Returns
        -------
        measDict : dict of dicts
            Measurements of the images.

        '''
        
        self.flatten_list()
        
        measDict = {}
        
        for i in self.images:
            
            #get matrix describing image
            imMatrix = io.imread(i)
            
            #preprocessing
            pp = PreProcessing(imMatrix)
            if pp.checkSize() == False:
                imMatrix = pp.rescaleImage()
            
            #retrieving name and movement of image
            path = i.split('.')
            paths = path[0].split('\\')
            name = paths[-1]
            movement = paths[0].split('/')[-1]
            
            measDict[name] = {}
            measDict[name]['movement'] = movement
            
            net = BuildNetwork(imMatrix, name, movement, R1, R2)
            SPmat = net.imageSegmentation(seg_type)
            
            if edge_type == 'Pixel Size':
                G = net.makeNetwork_pixelSize(SPmat)
                
            elif edge_type == 'Color':
                G = net.makeNetwork_color(SPmat)
                
            elif edge_type == 'Orientation':
                G = net.makeNetwork_orientation(SPmat)
                
            else:
                raise Exception("Invalid edge creation choice. Options are: 'Pixel Size', 'Color', 'Orientation', or 'Entropy'.")
                
            meas = Measurements(G, edge_type)
            imgMeas = meas.takeChosenMeasurements(meas_list)
            measDict[name].update(imgMeas)
            
        return measDict
    
    
    def dict_to_array(self, measDict, meas_list):
        '''

        Parameters
        ----------
        measDict : dict of dicts
            This describes the measurements for each image being analyzed.
        meas_list : list of str
            List used in networkAndMeasurement function.

        Returns
        -------
        measArray : array of floats
            The array version of the measurements, not including the name or 
            movement of the image. Same order as the dict. First index = image,
            second index = measurements

        '''
       
        measArray = np.zeros((len(measDict), len(meas_list)))
        
        a = 0
        for i in measDict: #for each painting
            b = 0
            for j in measDict[i]: # for each measurement
                if j != 'movement': # dont include the movement name in the array
                                    # of measurements
                    measArray[a][b] = measDict[i][j]
                    b += 1 #update measurement
            a += 1 #update painting
            
        return measArray
    
    
    def exportRawData(self, edge_type, seg_type, R1, R2, meas_list=None):
        '''

        Parameters
        ----------
        edge_type : str
            User choice of what ECT to use.
        seg_type : str
            User choice of what segmentation technique to use.
        meas_list : list of str
            Describes measurements being taken.

        Returns
        -------
        None.

        '''
        # you may want to reorganize this... this version is not parallelized
        if meas_list == None:
            #create network, take measurements, and convert the dict to an array
            measDict = self.networkAndMeasurement(edge_type, seg_type, self.meas_list, R1, R2)
            measArray = self.dict_to_array(measDict, self.meas_list)
        else:
            #create network, take measurements, and convert the dict to an array
            measDict = self.networkAndMeasurement(edge_type, seg_type, meas_list, R1, R2)
            measArray = self.dict_to_array(measDict, meas_list)
        
        
        if self.bootstrap == False:
            pd.DataFrame(measArray).to_csv(f'Measurements/Raw Data/{str(R1)} to {str(R2)}/{edge_type}_{seg_type}_raw.csv', 
                                           index=False, header=False, na_rep='0')
        
        else:
            pd.DataFrame(measArray).to_csv(f'Measurements/Raw Data/{str(R1)} to {str(R2)}/{edge_type}_{seg_type}_raw.csv', 
                                           index=False, header=False, na_rep='0')
