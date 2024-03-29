from imports import tf
from imports import tfds
from imports import os
from imports import time
from imports import nltk
from imports import np

def timer(start):
    print(f'{round(time.time() - start, ndigits = 4)}s have passed.')


def check_cwd():
    '''
    Checks if the current working directory is correct. Promts to correct, if otherwise.
    '''

    # check if current working directory is the repository, and set it if not
    cwd = os.getcwd()

    # for everyone but me, muhahaha
    while cwd[-22:] != 'Tensorflow_Project_NLP':
        print('The current working directory is not the top level folder of the repository "Tensorflow_Project_NLP".')
        print(f'You\'re currently here: {cwd}')
        new_path = input('Please navigate to the repository: ')
        
        try: os.chdir(new_path)
        except: print('This didn\'t work, please try again. \n\r')
        cwd = os.getcwd()

def get_data(buff_size = 1000, batch_size = 128):
    data = tfds.load('wikipedia')
    data = data['train']
    data = data.map(lambda x: x['text'], num_parallel_calls = tf.data.AUTOTUNE)

    # this sounds stupid, but we might want to sentenice our tokens
    # @tf.py_function(Tout = tf.string)
    # def make_sentences(x):
    #     return tf.constant(nltk.tokenize.sent_tokenize(x.numpy().decode('utf-8')))
    #     
    # data = data.map(lambda x: make_sentences(x), num_parallel_calls = tf.data.AUTOTUNE)
    
    data = data.shuffle(buff_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return data


# pad the data, so that they have equal sizes
@tf.py_function(Tout = tf.int32)
def pad_right(x, pad_len = 100, val = -1):
    
    # difference in padding between is and to be, can be negative
    pad_dim = pad_len.numpy() - x.shape[1]
        
    if pad_dim >= 0:
        # paddings = tf.constant([[0,0], [0, pad_dim]], dtype = tf.int32)
        x = tf.pad(x, [[0,0], [0, pad_dim]], 'CONSTANT', constant_values = val.numpy())
    else: # reduce tensor to match pad_len
        x = tf.slice(x, [0,0], size = [32,100])
        
        # prints for making sure stuff works
        # tf.print(f'Shape of Result: {x.shape}')
        # tf.print(f'Padding Dimension: {pad_dim}')

    return x


def targenise(text_data, tokeniser, max_tokens = 25000, padding = 50000, pad_val = -1):
    num_data = text_data.map(lambda x: tokeniser(x), num_parallel_calls = tf.data.AUTOTUNE)
    
    # some garbage collection
    del text_data
    del tokeniser
    
    num_data = num_data.map(lambda x: tf.cast(x, dtype = tf.int32))
    ######## that's the rolling idea of creating targets, resulting in output_layer needing to be len_pad long #####
    targets = num_data.map(lambda x: tf.roll(input = x, shift = -1, axis = 1), num_parallel_calls = tf.data.AUTOTUNE)
    num_data = num_data.map(lambda x: pad_right(x, padding, pad_val), num_parallel_calls = 1)
    targets  =  targets.map(lambda x: pad_right(x, padding, pad_val), num_parallel_calls = 1)

    # one hot, to create 3-dimensional targets
    targets = num_data.map(lambda x: tf.one_hot(x, max_tokens), num_parallel_calls = tf.data.AUTOTUNE)

    return num_data, targets

# get input tensor, and output the indices of the max values (basically undoing the one-hot encoding)
def make_readable(x):
    return tf.math.argmax(x, axis = 2, output_type = tf.int32)+1

# "undoing" the tokenisation
def string_from_token(x, vocab):
    vocab = np.asarray(vocab)
    return "".join(vocab[x.numpy()])

# given a string, predict using the model and the tokeniser
def generator(inputs, tokeniser, model):
    assert type(inputs) == str, f'This isn\'t a string, but rather {type(inputs)}'
    tokens = tokeniser(tf.constant([inputs], dtype = tf.string))
    x = pad_right(tokens)
    x = model(tokens)
    x = tf.math.argmax(x, axis = 2, output_type = tf.int32)
    vocab = np.asarray(tokeniser.layer.get_vocabulary())
    text = "".join(vocab[x.numpy()])
    

# function for getting sentences instead of full articles
def get_other_data(buff_size = 1000, batch_size = 128):
    data = tfds.load('wiki_auto/auto_full_with_split')
    data = data['full'] # HERE TAKE .take(10)
    data = data.map(lambda x: x['normal_sentence'], num_parallel_calls = tf.data.AUTOTUNE)
    # data = data.shuffle(buff_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return data