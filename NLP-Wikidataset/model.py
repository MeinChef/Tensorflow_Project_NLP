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
        
        
        # x = tf.keras.layers.Masking(mask_value = -1)(x) # masking doesn't work with CUDNN for some godforsaken reason
                
        for units in layer_units[1:]:
            x = tf.keras.layers.LSTM(units = units, return_sequences = True)(x, mask = mask)
        
        # x = tf.keras.layers.LSTM(units = layer_units[-1])(x)

        @tf.function
        def custom_soft(x):
            return tf.nn.softmax(x, axis = 2)

        outputs = tf.keras.layers.Dense(units = max_tokens, activation = custom_soft)(x)

        self.model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'Wikismart')
    
    @tf.function
    def call(self, x):
        return self.model(x)

    @tf.function
    def custom_soft(x):
        return tf.nn.softmax(x, axis = 2)

    def set_loss(self, loss = tf.keras.losses.CategoricalCrossentropy()):
        self.loss = loss
    
    def set_optimiser(self, optim = tf.keras.optimizers.Adam, learning_rate = 0.001):
        '''please pass the optimiser as an object, not a function (i.e. without the brackets -> `tf.keras.optimizers.Adam` )'''
        self.optim = optim(learning_rate = learning_rate)

    def set_metrics(self, loss_metr = tf.keras.metrics.Mean(name = 'loss'), acc_metr = tf.keras.metrics.CategoricalAccuracy(name = 'acc')):
        self.loss_metr = loss_metr
        self.acc_metr = acc_metr

    def lazy_setter(self):
        self.set_loss(tf.keras.losses.CategoricalCrossentropy(reduction = tf.keras.losses.Reduction.NONE))
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
            loss = self.loss(target, pred)

        #self.loss_metr.update_state(loss)
        self.acc_metr.update_state(target, pred)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))
    
    @tf.function
    def test(self, x, target):

        pred = self.model(x)
        loss = self.loss(target, pred)

        #self.loss_metr.update_state(loss)
        self.acc_metr.update_state(target, pred)

        return loss
    
    @tf.function
    def train_step(self, data, targets):

        with tf.GradientTape() as tape:
            predictions = self.model(data, training=True)
            per_example_loss = self.loss(targets, predictions)
            loss = tf.nn.compute_average_loss(per_example_loss)
            model_losses = self.model.losses
            if model_losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def training(self, data, targets, epochs):
        self.loss = np.empty(epochs)
        self.acc = np.empty(epochs)
        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            for x, t in tqdm(zip(data, targets)):
                self.train(x, t)
            

            self.loss[epoch], self.acc[epoch] = self.get_metrics()
            self.reset_metrics()
            
            # do we really need all the testing stuff?????
    
    @tf.function  
    def distributed_train_step(self, data, targets, strategy):
        per_replica_losses = strategy.run(self.train_step, args = (data, targets,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis = None)
    
    
    def distributed_training(self, data, targets, strategy, epochs):
        for epoch in range(epochs):
            for count, (x, t), in tqdm(enumerate(zip(data, targets))):
                print(self.distributed_train_step(x, t, strategy))
                if count % 10000: self.save_to_file(path = 'NLP-Wikidataset/model/LSTM/training_checkpoints')
            self.save_to_file(f'Epoch_{epoch}.keras')

    def info(self):
        return self.model.summary()


    #########################
    # methods for the model #
    #########################
    def save_to_file(self,name = None, path = None):
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