from imports import tf
from imports import tfds
from imports import os
from imports import np
from imports import tf_text
from imports import plt


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

def plot_acc_loss(data, specs, epochs):
    pass

    # assert data.shape == (8, epochs), f'Shape of the input data is off: expeced (8,10), got {data.shape}.'
    assert isinstance(specs, tuple), f'Expected tuple, got {type(specs)} instead.'
    
    x_ax = np.arange(1, epochs + 1)
    strs = []
    for struc, emb in zip(specs[0], specs[1]):
        strs.append(f'[{struc}] {emb}')
        
    assert len(strs) == len(specs[0])
        
    acc = (data[::2, :] * 100).T
    loss = data[1::2, :].T
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = True)
    
    fig.set_size_inches(10, 5)
        
    ax[0].plot(x_ax, acc,  label = strs)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy in %')
    ax[0].set_title('Accuracy of the different models')
    ax[0].legend(title = '[Layers] Embedding')
    
    
    ax[1].plot(x_ax, loss, label = strs)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Loss of the different models')
    ax[1].legend(title = '[Layers] Embedding')
    
    return fig, ax
    
    
        
    
    


# get input tensor, and output the indices of the max values (basically undoing the one-hot encoding)
def make_readable(x):
    return tf.math.argmax(x, axis = 2, output_type = tf.int32)

# "undoing" the tokenisation
def string_from_token(x, vocab):
    vocab = np.asarray(vocab)
    return "".join(vocab[x.numpy()])

# given a string, predict using the model and the tokeniser
def generator(inputs, tokeniser, model, length = 50,  pad_size = 264, pad_value = 0, search_depth = 3):
    assert type(inputs) == str, f'This isn\'t a string, but rather {type(inputs)}'
    assert search_depth > 0, f'Search depth is 0 or less than 0.'
    assert length > search_depth 
        
    # make tokens
    out = tokeniser(tf.constant([inputs], dtype = tf.string))
    out = out.to_list()[0]
        
    # selecting the ones with the highest probs
    for _ in range(length):
        
        with tf.device('GPU:1'):
            # padding
            x, _ = tf_text.pad_model_inputs(tf.ragged.constant([out]), pad_size, pad_value)
            
            # creating predictions
            x = model(x)
            
            # getting the top search_depth predictions
            _, new_tokens = tf.math.top_k(x, k = search_depth + 2, sorted = False)
            new_tokens = tf.squeeze(new_tokens)
            
            # filtering out the predictions for our current last token
            new_tokens = new_tokens[len(out)-1, :]
                
        

        
        # keep it from looping 3 words over and over again    
        for idx in range(search_depth):
            #                                          if the best fit wold be <UNK>, ignore it
            if new_tokens[idx] not in out[:-search_depth] and new_tokens[idx] != 1:
                out.append(new_tokens[idx].numpy())
                break
            
            # no fitting token was found:
            if idx == search_depth - 1:
                print(f'Found no token that\'s in the specified search depth of {search_depth}.\nResuming with nex best guess...')
                # check if this is now an <UNK> token
                if new_tokens[idx+1] != 1:
                    out.append(new_tokens[idx+1].numpy())
                else:
                    out.append(new_tokens[idx+2].numpy())
 
    vocab = np.asarray(tokeniser.layer.get_vocabulary())
    text = " ".join(vocab[np.asarray(out)])
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
