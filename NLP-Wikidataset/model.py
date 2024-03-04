from imports import tensorflow as tf

class LSTM(tf.keras.Model):
    def __init__(self, layer_units):
        
        # init inherited
        super().__init__()

        # specify structure of LSTM-Model
        if not isinstance(layer_units, (list, tuple)):
            layer_units = [layer_units]
        
        inputs = tf.keras.Input(shape = (), dtype = str) # still need to figure out shape

        for units in layer_units:
            x = tf.keras.layers.LSTM(units = units)(x)

    def call(self, x):
        self.model(x)

    @tf.function
    def train(self, x):
        pass
        
        x, target = x

        with tf.keras.GradientTape() as tape:
            pred = self.model(x)
            loss = self.loss(x, target)
        
        gradients = self.model()
        self.optimiser.apply_gradients(zip(self.model.trainable_variables, gradients))
    
    @tf.function
    def test(self, x):
        pass
        
        x, target = x

        pred = self.model(x)
        loss = self.loss(x, target)

        self.losses = loss

    def train_test():
        pass

    def info(self):
        return self.model.summary()

    def save(self, path):
        '''maybe this is deprecated'''
        try:
            self.model.save(path)
        except ValueError:
            print(f'Either Path: \'{path}\' is invalid or path is not of type str. (got {type(path)} instead)')

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