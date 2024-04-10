from imports import tf
from imports import np
from imports import re
from imports import os
from tqdm import tqdm

class LSTM(tf.keras.Model):
    def __init__(self, layer_units = [64, 64], embed_size = 74154, max_tokens = 50000, output_dim = 10):
        
        super().__init__()

        # specify structure of LSTM-Model
        if isinstance(layer_units, int):
            layer_units = [layer_units]
        
        inputs = tf.keras.Input(shape = (embed_size,), dtype = tf.uint16) 
        # x = tf.keras.layers.Embedding(input_dim = max_tokens, output_dim = output_dim)(inputs)
        self.embedding = tf.keras.layers.Embedding(input_dim = max_tokens, output_dim = output_dim)
        x = self.embedding(inputs)
        mask = self.embedding.compute_mask(inputs)
        
        x = tf.keras.layers.LSTM(units = layer_units[0], return_sequences = True)(x, mask = mask)
        for units in layer_units[1:]:
            x = tf.keras.layers.LSTM(units = units, return_sequences = True)(x)
        
        # x = tf.keras.layers.LSTM(units = layer_units[-1])(x)

        x = tf.keras.layers.Dense(units = max_tokens, activation = None)(x)
        outputs = tf.keras.layers.Softmax(axis = 2)(x)

        self.model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'Wikismart')
    
    @tf.function
    def call(self, x):
        return self.model(x)

    def set_loss(self, loss = tf.keras.losses.CategoricalCrossentropy()):
        self.loss_func = loss
    
    def set_optimiser(self, optim = tf.keras.optimizers.Adam, learning_rate = 0.001):
        '''please pass the optimiser as an object, not a function (i.e. without the brackets -> `tf.keras.optimizers.Adam` )'''
        self.optim = optim(learning_rate = learning_rate)

    def set_metrics(self, loss_metr = tf.keras.metrics.Mean(name = 'loss'), acc_metr = tf.keras.metrics.CategoricalAccuracy(name = 'acc')):
        self.loss_metr = loss_metr
        self.acc_metr = acc_metr

    def lazy_setter(self):
        self.set_loss(tf.keras.losses.SparseCategoricalCrossentropy(ignore_class = 0))
        self.set_metrics()
        self.set_optimiser()

    def reset_metrics(self): 
        self.loss_metr.reset_state()
        self.acc_metr.reset_state()

    def get_metrics(self):
        return self.loss_metr.result(), self.acc_metr.result()        
    
    @tf.function
    def train(self, x, target):
        
        with tf.GradientTape() as tape:
            pred = self.model(x)
            # is loss really usable?
            loss = self.loss_func(target, pred)

        self.loss_metr.update_state(loss)
        self.acc_metr.update_state(target, pred)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))

    def training(self, data, targets, epochs, extension = None, start = 0, text_file = None):
        self.loss = np.empty(epochs)
        self.acc = np.empty(epochs)
        
        train_epochs = epochs - start
        print(train_epochs, epochs, start)
        
        for epoch in range(train_epochs):
            print(f'Epoch {epoch + start}')
            for x, t in tqdm(zip(data, targets)):
                self.train(x, t)


            self.loss[epoch], self.acc[epoch] = self.get_metrics()
            self.reset_metrics()
            
            if extension: self.save_to_file(f'Sentence_Epoch{epoch + start}_{extension}', path = 'NLP-Wikidataset/model/LSTM/training_checkpoints')
            else: self.save_to_file(f'Sentence_Epoch{epoch}', path = 'NLP-Wikidataset/model/LSTM/training_checkpoints')
            
        if text_file:
            with open(text_file, 'a') as file:
                file.write(f'Epoch {epoch}:\n  Acc: {self.acc[epoch]}\n  Loss: {self.loss[epoch]}')
            
            
            

    def info(self):
        return self.model.summary()


    #########################
    # methods for the model #
    #########################
    def save_to_file(self, name = None, path = None):
        '''
        Saves the model as a .keras file.
        If no name given, saves it as model_{version}.keras. Else will save as {name}.keras
        '''
        if path == None: path = 'NLP-Wikidataset/model/LSTM'
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

            tf.print('Success saving the model!')
        
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



    def load_from_file(self, name = None, path = None):
        '''
        Loads a model with name as a .keras from NLP-Wikidataset/model/.
        If no name is given, uses latest save from the automatic version control.
        '''
        
        if path == None: path = 'NLP-Wikidataset/model/LSTM'

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