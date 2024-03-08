# from typing import Any
from imports import tf
from imports import os
from imports import re


class Tokeniser(tf.keras.Model):
    def __init__(self, max_tokens = 50000, output_mode = 'int', vocab = None):
        super().__init__()

        self.inputs = tf.keras.Input(shape = (1,), dtype = tf.string)
        self.layer = tf.keras.layers.TextVectorization(max_tokens = max_tokens, output_mode = output_mode, vocabulary = vocab)

    def call(self, vocab):
        self.model(vocab)

    def builder(self, name = 'Tokeniser'):
        outputs = self.layer(self.inputs)
        self.model = tf.keras.Model(inputs = self.inputs, outputs = outputs, name = name)
        self.model.compile()

    # methods for the layer, not the model, should be removed in final version
    def adapt(self, text):
        self.layer.adapt(text)

    def is_adapted(self):
        return self.layer.is_adapted()

    #########################
    # methods for the model #
    #########################
    def save_to_file(self, path = None):

        ### automatic version control: ###
        # get the contents of the '/model' folder
        if not path:
            contents_mofo = os.listdir('model')

            # get latest version
            versions = []
            for elem in contents_mofo:
                if re.match(r'model_\d+.keras', elem):
                    has_ver = re.search(r'\d+', elem).group()
                    versions.append(int(has_ver))

            if versions == []: ver = 0 # there is no model yet saved
            else: ver = max(versions) + 1
            
            # model path and saving
            path = f'model/model_{ver}.keras'
            self.model.save(path)

            print('Success saving the model!')
        
        ### custom save ##
        else:
            # check if path is valid
            if isinstance(path, (str, tf.string)):
                if path[-6:] == '.keras': 
                    # load model
                    self.model = tf.keras.models.load_model(path)

                else: raise ValueError(f'Path should end in \'.keras\'')
            else: raise ValueError(f'Expected string, got {type(path)} instead')



    def load_from_file(self, path = None):
        '''
        Loads a model from the specified path.
        If no path is given, uses latest save from the automatic version control.
        '''
        
        ### automatic version control: load latest model ###
        if not path:
            contents_mofo = os.listdir('model')

            # get latest version
            versions = []
            for elem in contents_mofo:
                if re.match(r'model_\d+.keras', elem):
                    has_ver = re.search(r'\d+', elem).group()
                    versions.append(int(has_ver))

            # load latest version
            self.model = tf.keras.models.load_model(f'model/model_{max(versions)}.keras')


        ### custom path ###
        else:
            # check if path is valid
            if isinstance(path, (str, tf.string)):
                if path[-6:] == '.keras': 
                    # load model
                    self.model = tf.keras.models.load_model(path)

                else: raise ValueError(f'Path should end in \'.keras\' and contain the model folder.')
            else: raise ValueError(f'Expected string, got {type(path)} instead')
            
            

        

