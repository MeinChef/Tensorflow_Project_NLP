from imports import tf
from imports import tfds
from imports import os
from imports import np
from imports import tf_text


def check_cwd():
    '''
    Checks if the current working directory is correct. Promts to correct, if otherwise.
    '''

    # check if current working directory is the repository, and set it if not
    cwd = os.getcwd()

    while cwd[-22:] != 'Tensorflow_Project_NLP':
        print('The current working directory is not the top level folder of the repository "Tensorflow_Project_NLP".')
        print(f'You\'re currently here: {cwd}')
        new_path = input('Please navigate to the repository: ')
        
        try: os.chdir(new_path)
        except: print('This didn\'t work, please try again. \n\r')
        cwd = os.getcwd()



def get_wiki_data(buff_size = 1000, batch_size = 128):
    data = tfds.load('wikipedia')
    data = data['train']
    data = data.map(lambda x: x['text'], num_parallel_calls = tf.data.AUTOTUNE)
    data = data.shuffle(buff_size).batch(batch_size, drop_remainder = True).prefetch(tf.data.AUTOTUNE)
    return data

# function for getting sentences instead of full articles
def get_sentence_data(buff_size = 1000, batch_size = 128):
    data = tfds.load('wiki_auto/auto_full_with_split')
    data = data['full']
    
    # we are only interested in the full sentences, but since this data is inteded to be used for a different task, we'll have to use the "normal sentence"
    data = data.map(lambda x: x['normal_sentence'], num_parallel_calls = tf.data.AUTOTUNE)
    data = data.shuffle(buff_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return data

# creating tokens and creating targets 
def targenise(text_data, tokeniser, max_tokens = 10000, padding = 264, pad_val = 0, batch_size = 64, buff_size = 1000):
    assert max_tokens <= tf.uint16.max, f'max_tokens ({max_tokens}) has a larger value than of uint16 ({tf.uint16.max})'
    
    # rolling/shifting doesn't work on ragged tensors, that's why the unbatching
    data = text_data.map(lambda x: tokeniser(x), num_parallel_calls = tf.data.AUTOTUNE).unbatch() # returns int64 and ragged tensors

    # some garbage collection
    del text_data
    del tokeniser
    
    # shifting the values one to the left to predict next word
    data = data.map(lambda x: \
                            (
                                x, 
                                tf.roll(input = x, shift = -1, axis = 0)
                            ),
                        num_parallel_calls = tf.data.AUTOTUNE
                    )
    
    # pad the data to the right shape
    data = data.map(lambda x, t: \
                            (
                                tf_text.pad_model_inputs(x, padding, pad_val),
                                tf_text.pad_model_inputs(t, padding, pad_val)
                            ),
                        num_parallel_calls = tf.data.AUTOTUNE
                    )
    
    # optimise resources
    data = data.map(lambda x, t: \
                            (
                                tf.cast(x[0], dtype = tf.uint16),
                                tf.cast(t[0], dtype = tf.uint16)
                            ),
                        num_parallel_calls = tf.data.AUTOTUNE
                    )

    data = data.shuffle(buff_size).batch(batch_size, drop_remainder = True).prefetch(tf.data.AUTOTUNE)
    return data


# get input tensor, and output the indices of the max values (basically undoing the one-hot encoding)
def make_readable(x):
    return tf.math.argmax(x, axis = 2, output_type = tf.int32)

# "undoing" the tokenisation
def string_from_token(x, vocab):
    vocab = np.asarray(vocab)
    return "".join(vocab[x.numpy()])

# given a string, predict using the model and the tokeniser
def generator(inputs, tokeniser, model, length = 50,  pad_size = 264, pad_value = 0):
    assert type(inputs) == str, f'This isn\'t a string, but rather {type(inputs)}'
    
    # make tokens
    tokens = tokeniser(tf.constant([inputs], dtype = tf.string))
    tokens = tf.cast(tokens, dtype = tf.int32)
    
    
    
    # selecting the ones with the highest probs
    for _ in range(length):
        # padding
        x, mask = tf_text.pad_model_inputs(tokens, pad_size, 0)
        # creating predictions
        x = model(x)
        _, indices = tf.math.top_k(x, k = 2)
        breakpoint()
    # tf.gather should be useful, according to the documentation, but I can't figure it out
    # z = tf.math.argmax(x, axis = 2, output_type = tf.int32)
    # y = tf.math.argmax(x, axis = 1, output_type = tf.int32)
    # w = tf.math.argmax(x, axis = 0, output_type = tf.int32)
    # make things readable
    vocab = np.asarray(tokeniser.layer.get_vocabulary())
    breakpoint()
    text = "".join(vocab[x.numpy()])
    return text



# this function throws errors from time to time. I do not know why. And it doesn't do it alwyas. 
# Sometimes it just goes over the entire dataset, other times it stops after the 3rd iteration. I'm at the end of my wits.
# @tf.py_function(Tout = tf.int32)
# def pad_right_old(x, pad_len = tf.constant(100, dtype = tf.int32), val = tf.constant(0, dtype = tf.int32), batch_size = tf.constant(64, dtype = tf.int32)):
    
#     # difference in padding between is and to be, can be negative
#     pad_dim = pad_len - x.shape[1]
        
#     if pad_dim >= 0:
#         try:
#             x = tf.pad(x, [[0,0], [0, pad_dim]], 'CONSTANT', constant_values = val.numpy())
#         except: raise ValueError(f'Shape of input: {x.shape}\nPad_dim: {pad_dim}')
        
#     else: # reduce tensor to match pad_len
#         x = tf.slice(x, [0,0], size = [batch_size, pad_len])
        
#     return x
