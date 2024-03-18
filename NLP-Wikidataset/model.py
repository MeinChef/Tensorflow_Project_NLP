from imports import tf
from imports import np
from imports import re
from imports import os
from tqdm import tqdm

class LSTM(tf.keras.Model):
    def __init__(self, layer_units = [64, 64], max_tokens = 50000):
        
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
        return self.call()
    
    def set_loss(self, loss = tf.keras.losses.CategoricalCrossentropy()):
        self.loss = loss
    
    def set_optimiser(self, optim = tf.keras.optimizers.Adam, learning_rate = 0.001):
        self.optim = optim(learning_rate = learning_rate)

    def set_metrics(self, loss_metr = tf.keras.metrics.Mean(name = 'loss'), acc_metr = tf.keras.metrics.CategoricalAccuracy(name = 'acc')):
        self.loss_metr = loss_metr
        self.acc_metr = acc_metr

    def reset_metrics(self): 
        self.loss_metr.reset_states()
        self.acc_metr.reset_states()

    def get_metrics(self):
        return self.loss_metr.result(), self.acc_metr.result()

    @tf.function
    def call(self, x):
        return self.model(x)

    @tf.function
    def train(self, x, target):
        
        with tf.keras.GradientTape() as tape:
            pred = self.model(x)
            # is loss really usable?
            loss = self.loss(x, target)

        self.loss_metr.update_state(loss)
        self.acc_metr.update_state(target, pred)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.model.trainable_variables))
    
    @tf.function
    def test(self, x, target):

        pred = self.model(x)
        loss = self.loss(x, target)

        self.loss_metr.update_state(loss)
        self.acc_metr.update_state(target, pred)

        return loss

    def train_test(self, data, targets, epochs):
        loss = np.empty()
        acc = np.empt()
        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            for x, t in tqdm(zip(data, targets)):
                self.train(x, t)

            loss[epoch], acc[epoch] = self.get_metrics()
            self.reset_metrics()
            
            # do we really need all the testing stuff?????


            
        

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
                    self.model.save(f'{path}/{name}.keras')
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