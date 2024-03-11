from imports import tf
from imports import tfds
from imports import os
from imports import time

class Timer():
    def __init__(self):
        self.start = time.time()
    
    def __call__(self):
        print(f'{time.time() - self.start}s have passed.')

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

def get_data(batch_size):
    data = tfds.load('wikipedia')
    data = data['train']
    data = data.map(lambda x: x['text'], num_parallel_calls = tf.data.AUTOTUNE)

    return data

def targenise(text_data, tokeniser):
    breakpoint()
    # isinstance(text_data, tf.data.Dataset)

    num_data = text_data.map(lambda x: tokeniser(x), num_parallel_calls = tf.data.AUTOTUNE)
    del text_data
    targets = num_data.map(lambda x: tf.roll(input = x, shift = -1, axis = 1), num_parallel_calls = tf.data.AUTOTUNE)
    return num_data, targets