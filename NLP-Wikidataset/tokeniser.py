from typing import Any
from imports import tf

class Tokeniser():
    def __init__(self, vocab = None):
        super().__init__()

        inputs = tf.keras.Input(shape = (1,), dtype = tf.string)

        if vocab:
            outputs = tf.keras.layers.TextVectorization(vocabulary = vocab)(inputs)
        else:
            outputs = tf.keras.layers.TextVectorization()(inputs)

        self.tokenise = tf.keras.Model(inputs = inputs, outputs = outputs)

        self.tokenisation_layer = tf.keras.layers.TextVectorization()

    def __call__(self, vocab):
        self.tokenise(vocab)

    # methods for the layer, not the model, should be removed in final version
    def adapt(self, text):
        self.tokenisation_layer.adapt(text)

    def is_adapted(self):
        return self.tokenisation_layer.is_adapted()
    
    # methods for the model
    def save_to_file(self, path = 'model/tokens.keras'):
        # check if path is valid
        if isinstance(path, (str, tf.string)):
            if path[-6:] == '.keras':
                # save model
                self.tokenise.save(path)

            else: raise ValueError('Path should end in \'.keras\'')
        else: raise ValueError(f'Expected string, got {type(path)} instead')

    def load_from_file(self, path):
        # check if path is valid
        if isinstance(path, (str, tf.string)):
            if path[-6:] == '.keras': 
                # load model
                self.tokenise = tf.keras.models.load_model(path)

            else: raise ValueError(f'Path should end in \'.keras\'')
        else: raise ValueError(f'Expected string, got {type(path)} instead')
        
        

        

