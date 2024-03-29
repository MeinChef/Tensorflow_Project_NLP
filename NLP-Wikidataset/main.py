from imports import tf
from imports import tfds
from imports import np
from imports import os
from imports import time
from imports import nltk 

import func
from tokeniser import Tokeniser
from model import LSTM

# tf.data.experimental.enable_debug_mode()

if __name__ == '__main__':

    #### split data 
    #### remove last non -null element to unk token
    
    # allows memory allocation, even if memory is not continuous
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    # some very interesting setup stuff
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')


    # if gpus:
    #     print("Num GPUs Available: ", len(gpus))
    #     try:
    #         strategy = tf.distribute.MirroredStrategy()
    #         # Allocate 6.5 GB of memory to each GPU
    #         for gpu in gpus:
    #             # tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit = 6656)])
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialised
    #         print(e)

    # ensures that the working directory is set correctly, to work with Tokeniser.save_to_file() and LSTM.save_to_file()
    func.check_cwd()




    # here the actual program starts: setting Hyperparameters
    EPOCHS = 1
    BATCH_SIZE = 128 # 416 this is the highest value that works with 74154 (longest wikipedia) and embedding of 1
    BUFFER_SIZE = 100
    MAX_TOKENS = 1000 # to use pre-tokenised model use 25k, 50k, 75k, or 100k
    PAD_SIZE = 11184
    # for the sentence dataset, mean sentence len is 140, 95 percentile is 264, 99 percentile 356
    # for the wikipedia dataset, mean article len is 3144, 95 percentile is 11184, 99 percentile 28025
    
    # code used for this: (unbatched)
    # i = 0
    # mean = np.empty(6672479)
    # for elem in raw_data:
    #     mean[i] = len(elem.numpy())
    #     i += 1

    # timer
    start = time.time()

    # get the data for the vocabulary
    raw_data = func.get_data(buff_size = BUFFER_SIZE, batch_size = BATCH_SIZE)

    tokeniser = Tokeniser(max_tokens = MAX_TOKENS)
    func.timer(start)

    ####### code to get tokens ######
    # with tf.device('/device:GPU:0'):
    #     tokeniser.adapt(raw_data)
    # tokeniser.builder()
    # tokeniser.save_layer(f'NLP-Wikidataset/model/layer/')
    # tokeniser.save_to_file(f'full_{int(MAX_TOKENS/1000)}k.keras')
    # tokeniser.save_to_file(f'sentence_dataset_{int(MAX_TOKENS/1000)}k_tokens.keras')
    
    ###### code to load pre-tokenised model ######
    tokeniser.load_layer(f'NLP-Wikidataset/model/layer/')
    # tokeniser.load_from_file(f'full_{int(MAX_TOKENS/1000)}k.keras')
    tokeniser.load_from_file(f'sentence_dataset_{int(MAX_TOKENS/1000)}k_tokens.keras')
    
    func.timer(start)
    
    
    num_data, targets = func.targenise(raw_data, tokeniser, max_tokens = MAX_TOKENS, padding = PAD_SIZE, pad_val = 0)
    del raw_data
    func.timer(start)
    

    model = LSTM(layer_units = [128, 128], embed_size = PAD_SIZE, max_tokens = MAX_TOKENS, output_dim = 20)
    model.lazy_setter()
    model.info()
    tf.print()

    model.train_test(data = num_data, targets = targets, epochs = EPOCHS)

    a = np.zeros((1,100), np.int32)
    a[0][0] = 15
    print(a)
    
    result = model(a)
    b = func.make_readable(result)
    
    try:
        d = func.string_from_token(b[0], tokeniser.layer.get_vocabulary())
    except:
        pass

    breakpoint()