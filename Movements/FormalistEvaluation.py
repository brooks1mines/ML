# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:25:58 2022

@author: khloe
"""

import pandas as pd
import glob
from skimage import io
import matplotlib.pyplot as plt

## Formalist Evaluation Script

class FormalistEvaluation():
    
    def __init__(self, movements, elementsDone=None):
        allImages = []
        for i in movements:
            
            #need all image types
            for filename in glob.glob('New Images/' + i + '/*.jpg'):
                allImages.append(filename)
                
            for filename in glob.glob('New Images/' + i + '/*.jpeg'):
                allImages.append(filename)
                
        self.allImages_filepaths = allImages
        
        if elementsDone == None:
            self.elementsDone = [0, 0, 0, 0, 0]
        else: 
            self.elementsDone = elementsDone
            
        # TODO: do self.oldchoices dictionaries as well...
        # TODO: make functions to check/redo chosen descriptor types
        
        # self.elementChoices = ['Shape and Space', 'Line, Form, and Value', 'Color', 'Texture', 'Other']
        # self.shapeChoices = {'Balance': ['Highly Balanced', 'Mostly Balanced', 'Mostly Imbalanced', 'Highly Imbalanced'],
        #                      'Movement': ['Highly Dynamic', 'Highly Static', 'Mix'],
        #                      'Empty Space': ['Large (>2/3)', 'Intermediate (>1/3)', 'Small (<1/3)']}
        # self.lineChoices = {'Depth': ['Flat', 'Shallow', 'Regular', 'Deep'],
        #                     'Local Value Contrast': ['Stark', 'Intermediate', 'Gradual', 'Mix'],
        #                     'Global Value Relationship': ['High', 'Intermediate', 'Low'],
        #                     'Average Value': ['Light', 'Balanced', 'Dark'],
        #                     'Lines/Object Differentiation': ['Distinct', 'Intermediate', 'Indistinct']}
        # self.colorChoices = {'Mode': ['Naturalistic', 'Capture of Conditions', 'Expressionistic'],
        #                      'Hue Diversity (Amount of Significant Colors)': ['Monochrome (1)', 'Similar (2-3)',
        #                                                              'Somewhat Diverse (4-5)', 'Highly Diverse(>5)'],
        #                      'Intensity': ['Vibrant', 'True', 'Muted/Tinted'],
        #                      'Global Color Relationship': ['Harmonious', 'Intermediate', 'Friction'],
        #                      'Local Color Contrast': ['Blended (Reality)', 'Intermediate (Small Blots of Color)',
        #                                               'Intermediate (Large Blots of Color)', 'Stark (Flat Change)']}
        # self.textureChoices = {'Brushstroke Control': ['Controlled', 'Mix', 'Wild'],
        #                        'Brushstroke Visibility': ['Visible', 'Fuzzy', 'Blended'],
        #                        'Outlines': ['Heavy', 'Intermediate', 'Light', 'Mix']}
        # self.otherChoices = {'Perspective': ['Removed', 'Intermediate', 'Direct', 'Abstract']}
    
        
    def __str__(self):
        return 'Formalist Evaluation Class'
    
    def whatElement(self):
        lookedAt = []
        notLookedAt = []
        for group in range(len(self.elementsDone)):
            index = group
            name = self.elementChoices[group]
            if self.elementsDone[group] != 0:
                lookedAt.append((index, name))
            else:
                notLookedAt.append((index, name))
            
        if len(lookedAt) == 0:
            print('You have not looked at any element groups yet. Your choices are:\n')
            choices = []
            for group in notLookedAt:
                print(f'{group[0]}. {group[1]}\n')
                choices.append(group[0])
                
            while True:
                try: 
                    elementChoice = int(input('What element group would you like to focus on?\n'))
                    assert elementChoice in choices
                except ValueError:
                    print('Enter an integer.')
                except AssertionError:
                    print(f'Choose one of the following: {choices}.')
                else:
                    break
        
                
        elif len(notLookedAt) == 0:
            print('You have looked at all possible element groups. ')
            elementChoice = -1
            
        else:    
            print('You have looked at the following element groups so far:\n')
            for group in lookedAt:
                print(f'{group[0]}. {group[1]}\n')
            print('Your choices are now:\n')
            
            choices = []
            for group in notLookedAt:
                print(f'{group[0]}. {group[1]}\n')
                choices.append(group[0])
                
            while True:
                try: 
                    elementChoice = int(input('What element group would you like to focus on?\n'))
                    assert elementChoice in choices
                except ValueError:
                    print('Enter an integer.')
                except AssertionError:
                    print(f'Choose one of the following: {choices}.')
                else:
                    break
        
        elementChoice = int(elementChoice)
        
        if elementChoice != -1:
            print(f'You have chosen {elementChoice}. {self.elementChoices[elementChoice]}.')
        else:
            print('Because of this, you can move on to other evaluations.\n')
            
        
        return elementChoice
    
    def shapeAndSpace(self):
        print('In terms of shape or space elements...\n')

        while True:
            try:
                balance = int(input('Space: 1. Highly Balanced, 2. Mostly Balanced, 3. Mostly Imbalanced, or 4. Highly Imbalanced?\n'))
                assert 0 < balance < 5
            except ValueError:
                print('Not an integer.\n')
            except AssertionError:
                print('Enter 1, 2, 3, or 4.\n')
            else:
                break
            
        while True:
            try:
                movement = int(input('Movement: 1. Highly Dynamic (Eye moves around whole composition),\n 2. Highly Static (eye focuses on one spot),\n or 3. Mix of Both (eye moves around in specific area of image)?\n'))
                assert 0 < movement < 4
            except ValueError:
                print('Not an integer.\n')
            except AssertionError:
                print('Enter 1, 2, or 3.\n')
            else:
                break
            
        while True:
            try:
                space = int(input('Space: 1. Large (>2/3), 2. Intermediate(<2/3, >1/3), or 3. Small empty space(<1/3)?\n'))
                assert 0 < space < 4
            except ValueError:
                print('Not an integer.\n')
            except AssertionError:
                print('Enter 1, 2, or 3.\n')
            else:
                break
            
        while True:
            try:
                perspective = int(input('Perspective: 1. Removed (wide crop), 2. Intermediate, or 3. Direct (tight crop)?\n'))
                assert 0 < perspective < 4
            except ValueError:
                print('Not an integer.')
            except AssertionError:
                print('Enter 1, 2, or 3.')
            else:
                break
        
        return balance, movement, space, perspective
    
    def lineFormAndValue(self):
        print('In terms of line, form, and value...\n')
        
        while True:
            try:
                depth = int(input('Depth: 1. Totally Flat, 2. Shallow, 3. Regular, or 4. Deep?\n'))
                assert 0 < depth < 5
            except ValueError:
                print('Not an integer.\n')
            except AssertionError:
                print('Enter 1, 2, 3, or 4.\n')
            else:
                break
            
        while True:
            try:
                contrast = int(input('Value Contrast (overall): 1. High, 2. Normal, or 3. Low?\n'))
                assert 0 < contrast < 4
            except ValueError:
                print('Not an integer.\n')
            except AssertionError:
                print('Enter 1, 2, or 3.\n')
            else:
                break
             
        while True:
            try:
                value = int(input('Average Value: 1. Light, 2. Dark, 3. Balanced?\n'))
                assert 0 < value < 4
            except ValueError:
                print('Not an integer.\n')
            except AssertionError:
                print('Enter 1, 2, or 3.\n')
            else:
                break
            
        while True:
            try:
                lines = int(input('Lines (between objects): 1. Distinct, 2. Intermediate, or 3. Indistinct?\n'))
                assert 0 < lines < 4
            except ValueError:
                print('Not an integer.')
            except AssertionError:
                print('Enter 1, 2, or 3.')
            else:
                break
      
        return depth, contrast, value, lines
    
    def color(self):
        print('In terms of color...\n')
        
        while True:
            try:
                mode = int(input('Mode: 1. Naturalistic, 2. Capture of Conditions, 3. Expressionistic, or 4. Other?\n'))
                assert 0 < mode < 5
            except ValueError:
                print('Not an integer.\n')
            except AssertionError:
                print('Enter 1, 2, 3, or 4.\n')
            else:
                break
            
        while True:
            try:
                diversity = int(input('Overall Diversity: 1. Monochrome (1), 2. Similar Hue (2-3), 3. Somewhat Diverse Hue(3-4), or 4. Highly Diverse Hue (>4)? \n'))
                assert 0 < diversity < 5
            except ValueError:
                print('Not an integer.\n')
            except AssertionError:
                print('Enter 1, 2, or 3.\n')
            else:
                break
            
        while True:
            try:
                intensity = int(input('Intensity (Tone): 1. Vibrant (little to no grey), 2. True, or 3. Muted (mostly greyed color)?\n'))
                assert 0 < intensity < 4
            except ValueError:
                print('Not an integer.\n')
            except AssertionError:
                print('Enter 1, 2, or 3.\n')
            else:
                break
            
            
        while True:
            try:
                harmony = int(input('Color Relationships: 1. Harmonious (spread spatial proximity), 2. Friction/Tension (smaller area(s) of distinct color), or 3. Other?\n'))
                assert 0 < harmony < 4
            except ValueError:
                print('Not an integer.\n')
            except AssertionError:
                print('Enter 1, 2, or 3.\n')
            else:
                break
            
        while True:
            try:
                contrast = int(input('Proximity Contrast: 1. Blended, 2. Stark, or 3. Other?\n'))
                assert 0 < contrast < 4
            except ValueError:
                print('Not an integer.\n')
            except AssertionError:
                print('Enter 1, 2, or 3.\n')
            else:
                break
        
        return mode, diversity, intensity, harmony, contrast
    
    def texture(self):
        print('In terms of texture...\n')
        
        while True:
            try:
                control = int(input('Brushstrokes: 1. Mostly Controlled, 2. Mix, or 3. Mostly Wild?\n'))
                assert 0 < control < 4
            except ValueError:
                print('Not an integer.\n')
            except AssertionError:
                print('Enter 1, 2, or 3.\n')
            else:
                break
            
        while True:
            try:
                visible = int(input('Brushstrokes: 1. Visible, 2. Blended, or 3. Mix?\n'))
                assert 0 < visible < 4
            except ValueError:
                print('Not an integer.\n')
            except AssertionError:
                print('Enter 1, 2, or 3.\n')
            else:
                break
            
        while True:
            try:
                outlines = int(input('Outlines: 1. Heavy, 2. Intermediate/Mix, or 3. Light?\n'))
                assert 0 < outlines < 4
            except ValueError:
                print('Not an integer.\n')
            except AssertionError:
                print('Enter 1, 2, or 3.\n')
            else:
                break
                
        return control, visible, outlines
    
    def doEvaluation(self):
        # get datafiles 
        shape_df = pd.read_csv('New Images/Formalism/Shape and Space.csv')
        line_df = pd.read_csv('New Images/Formalism/Line Form and Value.csv')
        color_df = pd.read_csv('New Images/Formalism/Color.csv')
        text_df = pd.read_csv('New Images/Formalism/Texture.csv')
        
        
        # get names of all images already analyzed for each element
        shape_names = shape_df['Name'].to_list()
        line_names = line_df['Name'].to_list()
        color_names = color_df['Name'].to_list()
        text_names = text_df['Name'].to_list()
        
        for image in self.allImages_filepaths:
            path = image.split('.')
            paths = path[0].split('\\')
            name = paths[-1]
            
            io.imshow(image)
            plt.title(name)
            plt.show()
            
            if name not in shape_names:
                    
                balance, movement, space, perspective = self.shapeAndSpace()
                shape_df.loc[len(shape_df.index)] = [name, balance, movement, space, perspective]
                    
                while True:
                    try:
                        contin = int(input('Continue? Press 0 to exit or 1 to continue.\n'))
                        assert -1 < contin < 2
                    except ValueError:
                        print('Input an integer.\n')
                    except AssertionError:
                        print('Enter 0 to exit or 1 to continue.\n')
                    else:
                        break
                
                if contin == 0:
                    break
            
            if name not in line_names:
                depth, contrast, value, lines = self.lineFormAndValue()
                line_df.loc[len(line_df.index)] = [name, depth, contrast, value, lines]
                
                while True:
                    try:
                        contin = int(input('Continue? Press 0 to exit or 1 to continue.\n'))
                        assert -1 < contin < 2
                    except ValueError:
                        print('Input an integer.\n')
                    except AssertionError:
                        print('Enter 0 to exit or 1 to continue.\n')
                    else:
                        break
                    
                if contin == 0:
                    break
           
            if name not in color_names:
                mode, diversity, intensity, harmony, contrast = self.color()
                color_df.loc[len(color_df.index)] = [name, mode, diversity, intensity, harmony, contrast]
                
                while True:
                    try:
                        contin = int(input('Continue? Press 0 to exit or 1 to continue.\n'))
                        assert -1 < contin < 2
                    except ValueError:
                        print('Input an integer.\n')
                    except AssertionError:
                        print('Enter 0 to exit or 1 to continue.\n')
                    else:
                        break
                
                if contin == 0:
                    break
                
            if name not in text_names:
                control, visibility, outlines = self.texture()
                text_df.loc[len(text_df.index)] = [name, control, visibility, outlines]
                
                while True:
                    try:
                        contin = int(input('Continue? Press 0 to exit or 1 to continue.\n'))
                        assert -1 < contin < 2
                    except ValueError:
                        print('Input an integer.\n')
                    except AssertionError:
                        print('Enter 0 to exit or 1 to continue.\n')
                    else:
                        break
                
                if contin == 0:
                    break
           
            
        shape_df.to_csv('New Images/Formalism/Shape and Space.csv', index=False)
        line_df.to_csv('New Images/Formalism/Line Form and Value.csv', index=False)
        color_df.to_csv('New Images/Formalism/Color.csv', index=False)
        text_df.to_csv('New Images/Formalism/Texture.csv', index=False)
        
        
        
if __name__ == '__main__':
    movements = ['Surrealism']
                  
    # movements_done = ['Baroque', 'Cubism', 'Expressionism', 'Fauvism', 'Impressionism', 'NeoClassicism', 
    #                   'PostImpressionism', 'Realism', 'Renaissance', 'Rococo', 'Romanticism', ]
    form = FormalistEvaluation(movements)
    form.doEvaluation()
    
        
    
        