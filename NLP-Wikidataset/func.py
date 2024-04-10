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

    # for everyone but me, muhahaha
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

    #  this sounds stupid, but we might want to sentenice our tokens
    # @tf.py_function(Tout = tf.string)
    # def make_sentences(x):
    #     return tf.constant(nltk.tokenize.sent_tokenize(x.numpy().decode('utf-8')))  
    # data = data.map(lambda x: make_sentences(x), num_parallel_calls = tf.data.AUTOTUNE)
    
    data = data.shuffle(buff_size).batch(batch_size, drop_remainder = True).prefetch(tf.data.AUTOTUNE)

    return data

# function for getting sentences instead of full articles
def get_sentence_data(buff_size = 1000, batch_size = 128):
    data = tfds.load('wiki_auto/auto_full_with_split')
    data = data['full'] # HERE TAKE .take(10)
    data = data.map(lambda x: x['normal_sentence'], num_parallel_calls = tf.data.AUTOTUNE)
    data = data.shuffle(buff_size).batch(batch_size, drop_remainder = True).prefetch(tf.data.AUTOTUNE)
    return data




# pad the data, so that they have equal sizes 

# this function throws errors from time to time. I do not know why. And it doesn't do it alwyas. 
# Sometimes it just goes over the entire dataset, other times it stops after the 3rd iteration. I'm at the end of my wits.

@tf.py_function(Tout = tf.int32)
def pad_right_old(x, pad_len = tf.constant(100, dtype = tf.int32), val = tf.constant(0, dtype = tf.int32), batch_size = tf.constant(64, dtype = tf.int32)):
    
    # difference in padding between is and to be, can be negative
    pad_dim = pad_len - x.shape[1]
        
    if pad_dim >= 0:
        try:
            x = tf.pad(x, [[0,0], [0, pad_dim]], 'CONSTANT', constant_values = val.numpy())
        except: raise ValueError(f'Shape of input: {x.shape}\nPad_dim: {pad_dim}')
        
    else: # reduce tensor to match pad_len
        x = tf.slice(x, [0,0], size = [batch_size, pad_len])
        
    return x

@tf.py_function(Tout = tf.int32)
def pad_right_new(x, pad_len = 264, val = 0, batch_size = 64):
    x = x.numpy()
    pad_dim = pad_len - x.shape[1]
    print(pad_dim)
    
    if pad_dim >= 0:
        x = np.pad(x, [(0,0),(0,pad_dim)], 'constant')
    else:
        x = x[:,:pad_len]
        print('HEY IT GOT REDUCED, DID IT DO THAT?')
        print(x.shape)
    
    return x
        
        
# pad right, but without the @tf.py_function decorator, gives more freedom
def pad_right(x, pad_len = 100, val = 0, batch_size = 64):
    
    # difference in padding between is and to be, can be negative
    pad_dim = pad_len - tf.shape(x)[1]
    breakpoint()
        
    if pad_dim >= 0:
        paddings = tf.constant([[0,0], [0, tf.get_static_value(pad_dim)]])
        x = tf.pad(x, paddings, 'CONSTANT', constant_values = val)
    else: # reduce tensor to match pad_len
        x = x[:,:pad_len]
            # x = tf.slice(x, [0,0], size = [batch_size, pad_len])
            # prints for making sure stuff works
            # tf.print(f'Shape of Result: {x.shape}')
            # tf.print(f'Padding Dimension: {pad_dim}')

    return x

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
                                tf_text.pad_model_inputs(t, padding, -1)
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
    
    # num_data = num_data.map(lambda x: pad_right(x, padding, pad_val, batch_size), num_parallel_calls = tf.data.AUTOTUNE)
    # targets  =  targets.map(lambda x: pad_right(x, padding, pad_val, batch_size), num_parallel_calls = tf.data.AUTOTUNE)

    # targets = targets.map(lambda x: tf_text.pad_model_inputs(x, padding, -1)) # padding with -1 results in one hot encoding not using 1s anywhere
    # targets = targets.map(lambda x, _: (tf.one_hot(x, max_tokens)), num_parallel_calls = tf.data.AUTOTUNE) # one-hot works only with: uint8, int8, int32, int64
    

    
    # num_data = num_data.map(lambda x: tf_text.pad_model_inputs(x, padding, pad_val))

    
    # num_data = num_data.map(lambda x, _: tf.cast(x, dtype = tf.uint16), num_parallel_calls = tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    # targets  =  targets.map(lambda x   : tf.cast(x, dtype = tf.uint16), num_parallel_calls = tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    # num_data = num_data.shuffle(buff_size).batch(batch_size, drop_remainder = True).prefetch(tf.data.AUTOTUNE)
    # targets  =  targets.shuffle(buff_size).batch(batch_size, drop_remainder = True).prefetch(tf.data.AUTOTUNE)

    return data





# get input tensor, and output the indices of the max values (basically undoing the one-hot encoding)
def make_readable(x):
    return tf.math.argmax(x, axis = 2, output_type = tf.int32)+1

# "undoing" the tokenisation
def string_from_token(x, vocab):
    vocab = np.asarray(vocab)
    return "".join(vocab[x.numpy()])

# given a string, predict using the model and the tokeniser
def generator(inputs, tokeniser, model, length = 50,  pad_size = 264):
    assert type(inputs) == str, f'This isn\'t a string, but rather {type(inputs)}'
    
    # make tokens
    tokens = tokeniser(tf.constant([inputs], dtype = tf.string))
    tokens = tf.cast(tokens, dtype = tf.int32)
    

    # selecting the ones with the highest probs
    for _ in range(length):
            # padding
        x = pad_right(tokens, pad_len = tf.constant(pad_size, dtype = tf.int32))
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