from imports import tf

import func
from tokeniser import Tokeniser
from model_sentence import LSTM



if __name__ == '__main__':
    EPOCHS = 16
    BATCH_SIZE = 64
    BUFFER_SIZE = 100
    MAX_TOKENS = 10000
    PAD_SIZE = 264
    
    raw_data = func.get_sentence_data(buff_size = BUFFER_SIZE, batch_size = BATCH_SIZE)

    tokeniser = Tokeniser(max_tokens = MAX_TOKENS)
    
    with tf.device('GPU:0'):
        tokeniser.adapt(raw_data)
    tokeniser.builder()

    num_data, targets = func.targenise(raw_data, tokeniser, max_tokens = MAX_TOKENS, padding = PAD_SIZE, pad_val = 0, batch_size = BATCH_SIZE)

    model = LSTM(layer_units = [64 for _ in range(6)], embed_size = PAD_SIZE, max_tokens = MAX_TOKENS, output_dim = 192)
    model.lazy_setter()
    model.info()
    tf.print()
    
    with tf.device('GPU:0'):
        model.training(num_data, targets, 1)