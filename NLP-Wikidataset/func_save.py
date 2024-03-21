from imports import tf
from imports import tfds
from imports import os
from imports import time

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

def get_data(buff_size = 1000, batch_size = 128, padding = None, pad_val = -1):
    data = tfds.load('wikipedia')
    data = data['train']
    data = data.map(lambda x: x['text'], num_parallel_calls = tf.data.AUTOTUNE)
    data = data.shuffle(buff_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return data

def targenise(text_data, tokeniser, max_tokens = 25000, padding = 50000, pad_val = -1):
    num_data = text_data.map(lambda x: tokeniser(x), num_parallel_calls = tf.data.AUTOTUNE)
    del text_data
    num_data = num_data.map(lambda x: tf.cast(x, dtype = tf.int32))
    
    ######## that's the rolling idea of creating targets, resulting in output_layer needing to be len_pad long #####
    # targets = num_data.map(lambda x: tf.roll(input = x, shift = -1, axis = 1), num_parallel_calls = tf.data.AUTOTUNE)
    
    # pad the data, so that they have equal sizes
    # @tf.py_function(Tout = tf.int32)
    # def pad_right(x, pad_len, val = -1):
    #     paddings = tf.constant([[0,0], [0, pad_len.numpy() - x.shape[1]]], dtype = tf.int32)
    #     return tf.pad(x, paddings, 'CONSTANT', constant_values = val.numpy())
    #     
    # 
    # num_data = num_data.map(lambda x: pad_right(x, padding, pad_val), num_parallel_calls = tf.data.AUTOTUNE)
    # targets  =  targets.map(lambda x: pad_right(x, padding, pad_val), num_parallel_calls = tf.data.AUTOTUNE)

    ###### that's the one-hot, reducing idea of creating targets ######
    targets = num_data.map(lambda x: tf.one_hot(x, max_tokens), num_parallel_calls = tf.data.AUTOTUNE)
    targets = targets.map(lambda x: tf.reduce_max(x, axis = 1), num_parallel_calls = tf.data.AUTOTUNE)

    return num_data, targets


