from imports import tf
from imports import tfds
from tokeniser import Tokeniser


def load_and_prep_dataset(batch_size):
    
    data = tfds.load('wikipedia', shuffle_files = True)

    #  #### text = [example['text'].numpy().decode('utf-8') for example in text['train']] # this step seems utterly ridicolous, converting a dataset into a list???? like wtf
    # let's not do that, RAM overflow :)

    text = data['train']
    del data

    text = text.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    tokenise = tf.keras.layers.TextVectorization()

    tokenise.adapt(text.map(lambda words, label: words))




    
    


    # test = ['word', 'foo', 'bar', 'taz']
    # for body, title in text:
    #     tokenise(body)
# 
    # print(tokenise.get_vocabulary())
    
    
    #train = train.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    #test  =  test.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return 


# use the old tokeniser instead of the new layer, maybe that works better :MaiShrug:
def load_and_tokenise(batch_size):
    text = tfds.load('wikipedia', shuffle_files = True)

    tokeniser = tf.keras.preprocessing.text.Tokenizer()



#   # do we want to get chars or rather words?????
#       # I concluded this is uneccesary.
#   def id_from_chr(char):
#       return tf.keras.layers.StringLookup(vocabulary = list(vocab), mask_token = None)(char)
#   
#   def char_from_id(id):
#       return tf.keras.layers.StringLookup(vocabulary = list(vocab), invert = True, mask_token = None)(id)
#   
#   def text_from_ids(ids):
#       return tf.strings.reduce_join(char_from_id(ids), axis = -1)

# rather, we want to train a model on "vocabulary"
# see tokenise

if __name__ == '__main__':

    BATCH_SIZE = 4

    a = load_and_prep_dataset(BATCH_SIZE)
    print(type(a))
