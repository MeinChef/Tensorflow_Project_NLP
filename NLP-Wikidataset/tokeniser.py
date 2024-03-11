# from typing import Any
from imports import tf
from imports import os
from imports import re


class Tokeniser(tf.keras.Model):
    def __init__(self, max_tokens = 50000, output_mode = 'int', vocab = None):
        super().__init__()

        self.inputs = tf.keras.Input(shape = (1,), dtype = tf.string)
        self.layer = tf.keras.layers.TextVectorization(max_tokens = max_tokens, output_mode = output_mode, vocabulary = vocab)


    @tf.function
    def call(self, x):
        return self.model(x)

    

    def builder(self, name = 'Tokeniser'):
        outputs = self.layer(self.inputs)
        self.model = tf.keras.Model(inputs = self.inputs, outputs = outputs, name = name)
        self.model.compile()

    def info(self):
        return self.model.summary()

    # methods for the layer, not the model, should be removed in final version
    def adapt(self, text):
        self.layer.adapt(text)

    def is_adapted(self):
        return self.layer.is_adapted()

    #########################
    # methods for the model #
    #########################
    def save_to_file(self, name = None):
        '''
        Saves the model as a .keras file.
        If no name given, saves it as model_{version}.keras. Else will save as {name}.keras
        '''
        path = 'NLP-Wikidataset/model/Tokenise'
        ### automatic version control: ###
        # get the contents of the 'NLP-Wikidataset/model' folder
        if name == None:
            contents_mofo = os.listdir(path)

            # get latest version
            versions = []
            for elem in contents_mofo:
                if re.match(r'model_\d+.keras', elem):
                    ver = re.search(r'\d+', elem).group()
                    versions.append(int(ver))

            if versions == []: ver = 0 # there is no model yet saved
            else: ver = max(versions) + 1
            
            # model path and saving
            path = f'{path}/model_{ver}.keras'
            self.model.save(path)

            print('Success saving the model!')
        
        ### custom save ##
        else:
            # check if path is valid
            # maybe replace the if isinstnace with assert
            if isinstance(name, (str, tf.string)):
                if name[-6:] == '.keras':
                    name = re.sub(r'\.|/', '', name[:-6])
                    self.model.save(f'{path}/{name}')
                else:
                    name = re.sub(r'\.|/', '', name)
                    self.model.save(f'{path}/{name}.keras')

            else: raise ValueError(f'Expected string, got {type(path)} instead')



    def load_from_file(self, name = None):
        '''
        Loads a model with name as a .keras from NLP-Wikidataset/model/.
        If no name is given, uses latest save from the automatic version control.
        '''
        
        path = 'NLP-Wikidataset/model/Tokenise'

        ### automatic version control: load latest model ###
        if name == None:
            contents_mofo = os.listdir(path)

            # get latest version
            versions = []
            for elem in contents_mofo:
                if re.match(r'model_\d+.keras', elem):
                    ver = re.search(r'\d+', elem).group()
                    versions.append(int(ver))

            # load latest version
            self.model = tf.keras.models.load_model(f'{path}/model_{max(versions)}.keras')


        ### custom model name ###
        else:
            # check if path is valid
            if isinstance(name, (str, tf.string)):
                if name[-6:] == '.keras':
                    self.model = tf.keras.models.load_model(f'{path}/{name}')
                else: 
                    self.model = tf.keras.models.load_model(f'{path}/{name}.keras')

            else: raise ValueError(f'Expected string, got {type(name)} instead')
            
            

        

