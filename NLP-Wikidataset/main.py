from imports import tf
from imports import np
from imports import os

import func
from tokeniser import Tokeniser
from model import LSTM


# for the sentence dataset, mean sentence len is 140, 95 percentile is 264, 99 percentile 356
# for the wikipedia dataset, mean article len is 3144, median 1421, 75 percentile 3293, 80 percentile is 4038, 85 percentile is 5139 (OOM), 90 percentile is 6988 (OOM), 95 percentile is 11184 (gives OOM error), 99 percentile 28025, max is 463680

# code used for this: (unbatched data)
# i = 0
# mean = np.empty(6672479)
# for elem in raw_data:
#     mean[i] = len(elem.numpy())
#     i += 1
# breakpoint() # execute here: np.median(mean), np.percentile(mean, perc), np.mean(mean)


if __name__ == '__main__':

    #### split data 
    #### remove last non -null element to unk token
    #### custom loss
    #### checkpoints
    
    # allows memory allocation, even if memory is not continuous
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    # some very interesting setup stuff
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')


    if gpus:
        print("Num GPUs Available: ", len(gpus))
        try:
            strategy = tf.distribute.MirroredStrategy()
            for gpu in gpus:
                # Allocate 6.5 GB of memory to each GPU
                tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit = 7168)])
                
                # set memory_growth = True 
                # tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialised
            print(e)

    # ensures that the working directory is set correctly, to work with Tokeniser.save_to_file() and LSTM.save_to_file()
    func.check_cwd()



    # here the actual program starts: setting Hyperparameters
    EPOCHS = 16
    BATCH_SIZE_PER_WORKER = 14 # 15 seems to be the sweetspot for 2500 Tokens, 3000 pad size and 10 dims
                    # 14 for same parameters, output dim = 20 to 40
    BATCH_SIZE = BATCH_SIZE_PER_WORKER * strategy.num_replicas_in_sync
    BUFFER_SIZE = 100
    MAX_TOKENS = 2500
    PAD_SIZE = 3000

    # since we're using uint16, for memory reasons, we must make sure max tokens doesn't exceed the max value of uint16
    if MAX_TOKENS > tf.uint16.max: raise ValueError(f'The variable \'MAX_TOKENS\' (value: {MAX_TOKENS}) exceeds the maximum value of a uint16 ({tf.uint16.max}).')

    # get the data for the vocabulary
    raw_data = func.get_data(buff_size = BUFFER_SIZE, batch_size = BATCH_SIZE)
    # initialise the Tokeniser
    tokeniser = Tokeniser(max_tokens = MAX_TOKENS)

    ####### code to get tokeniser ######
    # with tf.device('GPU:0'):
    #     tokeniser.adapt(raw_data)
    # tokeniser.builder()
    # tokeniser.save_layer(f'NLP-Wikidataset/model/layer/{MAX_TOKENS}_tokens/')
    # tokeniser.save_to_file(f'full_{int(MAX_TOKENS/1000)}k.keras')
    # tokeniser.save_to_file(f'sentence_dataset_{int(MAX_TOKENS/1000)}k_tokens.keras')
    
    ###### code to load pre-tokenised model ######
    tokeniser.load_layer(f'NLP-Wikidataset/model/layer/{MAX_TOKENS}_tokens/')
    tokeniser.load_from_file(f'full_{int(MAX_TOKENS/1000)}k.keras')
    # tokeniser.load_from_file(f'sentence_dataset_{int(MAX_TOKENS/1000)}k_tokens.keras')
            
    num_data, targets = func.targenise(raw_data, tokeniser, max_tokens = MAX_TOKENS, padding = PAD_SIZE, pad_val = 0, batch_size = BATCH_SIZE)
    del raw_data    
    
    num_data_dist = strategy.experimental_distribute_dataset(num_data)
    targets_dist = strategy.experimental_distribute_dataset(targets)
    
    # Create a checkpoint directory to store the checkpoints.
    checkpoint_dir = './NLP-Wikidataset/model/LSTM/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


    # this needs to happen within the strategy.scope()
    with strategy.scope():
        model = LSTM(layer_units = [128, 128], embed_size = PAD_SIZE, max_tokens = MAX_TOKENS, output_dim = 40)
        model.lazy_setter()
        model.info()
        tf.print()
        
        model.distributed_training(num_data, targets, strategy)
        # model.training(data = num_data, targets = targets, epochs = EPOCHS)
        model.save_to_file('trained_wiki')
    breakpoint()

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