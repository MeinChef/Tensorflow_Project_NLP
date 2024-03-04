# from typing import Any
from imports import tf
from imports import os


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

    # methods for the layer, not the model, should be removed in final version
    def adapt(self, text):
        self.layer.adapt(text)

    def is_adapted(self):
        return self.layer.is_adapted()
    
    # methods for the model
    def save_to_file(self, path = '/model/tokens.keras'):
        # check if path is valid
        if isinstance(path, (str, tf.string)):
            if path[-6:] == '.keras':
                # save model
                self.model.save(os.path.dirname(path)) # os.path.dirname(path)

            else: raise ValueError('Path should end in \'.keras\'')
        else: raise ValueError(f'Expected string, got {type(path)} instead')

    def load_from_file(self, path):
        # check if path is valid
        if isinstance(path, (str, tf.string)):
            if path[-6:] == '.keras': 
                # load model
                self.model = tf.keras.models.load_model(path)

            else: raise ValueError(f'Path should end in \'.keras\'')
        else: raise ValueError(f'Expected string, got {type(path)} instead')
        
        

        

