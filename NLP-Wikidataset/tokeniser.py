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
        self.tokenise_try = self.tokenise(vocab)

    def adapt_it(self, text):
        self.tokenisation_layer.adapt(text)
        
        

        

