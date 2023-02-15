# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:52:46 2022

@author: khloe
"""

import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import glob

## CRITICAL EVALUATION
class CriticalEvaluation():
    
    def __init__(self, movements):
        allImages = []
        for i in movements:
            
            #need all image types
            for filename in glob.glob('Movements/New Images/' + i + '/*.jpg'):
                allImages.append(filename)
                
            for filename in glob.glob('Movements/New Images/' + i + '/*.jpeg'):
                allImages.append(filename)
                
        self.allImages_filepaths = allImages
        
    def __str__(self):
        return 'Content Critical Evaluation Class'
    
    def genrePainting(self):
        print('What is happening in the image?\n')
        
        while True:
            try:
                location = int(input('Location: 1. Indoors or 2. Outdoors\n'))
                assert 0 < location < 3
            except ValueError:
                print('Input an integer.\n')
            except AssertionError:
                print('Input 1 or 2.\n')
            else:
                break
            
        while True:
            try:
                domain = int(input('Domain: 1. Public or 2. Domestic\n'))
                assert 0 < domain < 3
            except ValueError:
                print('Input an integer.\n')
            except AssertionError:
                print('Input 1 or 2.\n')
            else:
                break
            
        while True:
            try:
                realm = int(input('Realm: 1. Urban, 2. Rural, or 3. Other/Unsure\n'))
                assert 0 < realm < 4
            except ValueError:
                print('Input an integer.\n')
            except AssertionError:
                print('Input 1, 2, or 3.\n')
            else:
                break
        
        print('Action Options:\n1. Recreation\n2. Rest/Leisure\n3. Work\n4. Dining\n5. Religious\n6. Education')
        print('7. Chores\n8. Festivity\n9. Violence\n10. Conflict\n11. Communication\n12. Other\n')
        
        actions = []
        while True:
            while True:
                try:
                    action = int(input('Action:\n'))
                    assert action not in actions and -1 < action < 13
                except ValueError:
                    print('Input an integer.\n')
                except AssertionError:
                    print('Input a number between 0 and 12.\n')
                else:
                    break
            if action == 0:
                break
            else:
                actions.append(action)
                print('Enter 0 to exit.')
         
        print('What is the message of the image? The options are:\n')
        print('1. Community/Family\n2. Spirituality\n3. Cycle of life\n4. Science\n5. Government\n6. War')
        print('7. Social Consciousness\n8. the Body\n9. the Mind\n10. Power\n11. Beauty/Ugly\n12. Love')
        print('13. Morality\n14. Occupation of Time\n15. Other\n')
        
        messages = []
        while True:
            while True:
                try:
                    message = int(input('Message:\n'))
                    assert message not in messages and -1 < message < 16
                except ValueError:
                    print('Input an integer.\n')
                except AssertionError:
                    print('Input a number between 0 and 15.\n')
                else:
                    break
            if message == 0:
                break
            else:
                if message == 15:
                    descr = input('Describe the other descriptor.\n')
                    messages.append(descr)
                else:
                    messages.append(message)
                
                print('Enter 0 to exit.')
                
        return location, domain, realm, actions, messages
    
    def historyPainting(self):
        print('What subgenre is the image? Your options are:\n')
        print('1. Historical\n2. Religious\n3. Mythological/Folklore\n4. Literary\n5. Allegorical\n')
        
        while True:
            try:
                subgenre = int(input('Subgenre:\n'))
                assert 0 < subgenre < 6
            except ValueError:
                print('Input an integer.\n')
            except AssertionError:
                print('Input a number between 1 and 5.\n')
            else:
                break
            
        location, domain, realm, actions, messages = self.genrePainting()
        
        return subgenre, location, domain, realm, actions, messages
    
    def portraiturePainting(self):
        print('Who is in this image? The options are:\n')
        print('1. Royalty\n2. Government/Military\n3. Aristocracy\n4. Commoner\n5. Thinker/Artist\n6. Religious\n7. Other/Self/Personal\n')
        
        while True:
            try:
                person = int(input('Person:\n'))
                assert 0 < person < 8
            except ValueError:
                print('Input an integer.\n')
            except AssertionError:
                print('Input a number between 1 and 7.\n')
            else:
                break
            
        print('What is the message? The options are:\n')
        print('1. Status\n2. Power\n3. Genius\n4. Beauty/Ideal\n5. Strife/Pain\n6. Exhaustion\n7. Divinity/Greatness')
        print('8. Joyous\n9. Delicacy/Youth/Sensitivity\n10. Naivety\n11. Serious\n12. Dedication\n13. Sadness')
        print('14. Mischievous\n15. Thoughtful\n16. Other\n')
        
        messages = []
        while True:
            while True:
                try:
                    message = int(input('Message:\n'))
                    assert message not in messages and -1 < message < 17
                except ValueError:
                    print('Input an integer.\n')
                except AssertionError:
                    print('Input a number between 0 and 15.\n')
                else:
                    break
            if message == 0:
                break
            else:
                if message == 16:
                    descr = input('Describe the other descriptor.\n')
                    messages.append(descr)
                else:
                    messages.append(message)
                
                print('Enter 0 to exit.')
                
        return person, messages
    
    def landscapePainting(self):
        print('Where is the image? The options are:\n')
        print('1. Farm/grassland\n2. Mountain\n3. Sea/Ocean\n4. Forest\n5. River/Lake\n6. City\n')
        print('7. Beach\n')
        
        while True:
            try: 
                location = int(input('Location:\n'))
                assert 0 < location < 8
            except ValueError:
                print('Input an integer.\n')
            except AssertionError:
                print('Input a number between 1 and 7.\n')
            else:
                break
        
        print('What else is included in the image? The options are:\n')
        print('1. Animals\n2. Humans\n3. Buildings\n4. Other\n5. None\n')
        
        while True:
            try:
                objects = int(input('Objects:\n'))
                assert 0 < objects < 6
            except ValueError:
                print('Input an integer.\n')
            except AssertionError:
                print('Input a number between 1 and 5.\n')
            else:
                break
        
        print('What is the message? The options are:\n')
        print('1. Power\n2. Peaceful\n3. Beauty/Ugly\n4. Mystery\n5. Danger\n')
        print('6. Bountiful/Giving\n7. Unforgiving\n8. Other\n')
        
        messages = []
        while True:
            while True:
                try:
                    message = int(input('Message:\n'))
                    assert message not in messages and -1 < message < 9
                except ValueError:
                    print('Input an integer.\n')
                except AssertionError:
                    print('Input a number between 0 and 9.\n')
                else:
                    break
            if message == 0:
                break
            else:
                if message == 8:
                    description = input('Describe the other descriptor.\n')
                    messages.append(description)
                else:
                    messages.append(message)
                print('Enter 0 to exit.')
                
        return location, objects, messages
    
    def stillLifePainting(self):
        print('What objects are in the image? The options are:\n')
        print('1. Food\n2. Furniture\n3. Tools\n4. Flowers/Plants\n5. Other\n')
        
        objects = []
        while True:
            while True:
                try:
                    objectt = int(input('Objects:\n'))
                    assert objectt not in objects and -1 < objectt < 8
                except ValueError:
                    print('Input an integer.\n')
                except AssertionError:
                    print('Input a number between 0 and 6.\n')
                else:
                    break
            if objectt == 0:
                break
            else:
                objects.append(objectt)
                print('Enter 0 to exit.')
                
        return objects
    
    def abstractionAndGenre(self):
        print('What level of abstraction is the image? The options are:\n')
        print('1. Wholly abstract - no resemblance to natural shapes\n')
        print('2. Organically abstract - some resemblance to natural organic forms\n')
        print('3. Semi-abstract - figures and other objects are discernable to an extent\n')
        print('4. Naturalistic - figurative and other content is instantly recognizable\n')
        
        while True:
            try:
                abstraction = int(input('Level of Abstraction:\n'))
                assert 0 < abstraction < 5
            except ValueError:
                print('Input an integer.\n')
            except AssertionError:
                print('Input a number between 1 and 4.\n')
            else:
                break
        
        if abstraction != 1:
            print('What is the genre?\n')
            while True:
                try:
                    genre = int(input('Genre: 1. Historical, 2. Portraiture, 3. Genre, 4. Landscape, or 5. Still Life\n'))
                    assert 0 < genre < 6
                except ValueError:
                    print('Input an integer.\n')
                except AssertionError:
                    print('Input a number between 1 and 5.\n')
                else:
                    break
                
            if genre == 1:
                subgenre, location, domain, realm, actions, messages = self.historyPainting()
                output = [subgenre, location, domain, realm, actions, messages]
                
            elif genre == 2:
                person, messages = self.portraiturePainting()
                output = [person, messages]
                
            elif genre == 3:
                location, domain, realm, actions, messages = self.genrePainting()
                output = [location, domain, realm, actions, messages]
                
            elif genre == 4:
                location, objects, messages = self.landscapePainting()
                output = [location, objects, messages]
                
            else:
                objects = self.stillLifePainting()
                output = [objects]
                
        else:
            genre = None
            output = [None]
            
        return abstraction, genre, output
        
    
    def doEvaluation(self):
        abstract_df = pd.read_csv('Movements/New Images/Critical Theory/Abstraction Level.csv')
        genre_df = pd.read_csv('Movements/New Images/Critical Theory/Genres.csv')
        outputs_df = pd.read_csv('Movements/New Images/Critical Theory/Descriptors.csv')
        
        names = abstract_df['Name'].to_list()
        
        for image in self.allImages_filepaths:
            path = image.split('.')
            paths = path[0].split('\\')
            name = paths[-1]
            
            if name not in names:
                io.imshow(image)
                plt.title(name)
                plt.show()
            
                abstraction, genre, output = self.abstractionAndGenre()
                
                abstract_df.loc[len(abstract_df.index)] = [name, abstraction]
                genre_df.loc[len(genre_df.index)] = [name, genre]
                
                while len(output) < 6:
                    output.append(None)
                
                new_output = [name]
                for i in output:
                    new_output.append(i)
                    
                outputs_df.loc[len(outputs_df.index)] = new_output
                
                while True:
                    try:
                        contin = int(input('Continue? Press 0 to exit or 1 to continue.\n'))
                        assert -1 < contin < 2
                    except ValueError:
                        print('Input an integer.')
                    except AssertionError:
                        print('Enter 0 to exit or 1 to continue.')
                    else:
                        break
                    
            else:
                contin = 1
                
            if contin == 0:
                break
                
        abstract_df.to_csv('Movements/New Images/Critical Theory/Abstraction Level.csv', index=False)
        genre_df.to_csv('Movements/New Images/Critical Theory/Genres.csv', index=False)
        outputs_df.to_csv('Movements/New Images/Critical Theory/Descriptors.csv', index=False)
        
if __name__ == '__main__':
    movements = ['Surrealism']
    # movements_done = ['Baroque', 'Cubism', 'Expressionism', 'Fauvism', 'Impressionism', 
    #                   'NeoClassicism', 'PostImpressionism', 'Realism', 'Renaissance', 'Rococo', 'Romanticism',]
    crit = CriticalEvaluation(movements)
    crit.doEvaluation()
                
        
            
    