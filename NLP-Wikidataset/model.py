from imports import tf
from imports import re
from imports import os

class LSTM(tf.keras.Model):
    def __init__(self, layer_units, max_tokens):
        
        super().__init__()

        # specify structure of LSTM-Model
        if isinstance(layer_units, int):
            layer_units = [layer_units]
        
        inputs = tf.keras.Input(shape = (), dtype = tf.int32) # still need to figure out shape
        
        x = tf.keras.layers.Embedding(input_dim = max_tokens, output_dim = 10)(inputs)

        for units in layer_units:
            x = tf.keras.layers.LSTM(units = units)(x)

        outputs = tf.keras.layers.Dense(units = max_tokens, activation = tf.nn.softmax)

        self.model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'Wikismart')

    def __call__(self, x):
        self.call()

    @tf.function
    def call(self, x):
        self.model(x)

    @tf.function
    def train(self, x):
        
        x, target = x

        with tf.keras.GradientTape() as tape:
            pred = self.model(x)
            loss = self.loss(x, target)
        
        gradients = self.model()
        self.optimiser.apply_gradients(zip(self.model.trainable_variables, gradients))
    
    @tf.function
    def test(self, x):
        
        x, target = x

        pred = self.model(x)
        loss = self.loss(x, target)

        self.losses = loss

    def train_test():
        pass

    def info(self):
        return self.model.summary()


    #########################
    # methods for the model #
    #########################
    def save_to_file(self, name = None):
        '''
        Saves the model as a .keras file.
        If no name given, saves it as model_{version}.keras. Else will save as {name}.keras
        '''
        path = 'NLP-Wikidataset/model/LSTM'
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
        
        path = 'NLP-Wikidataset/model/LSTM'

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