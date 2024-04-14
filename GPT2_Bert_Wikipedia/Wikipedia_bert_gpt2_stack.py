import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import pipeline, set_seed, TFAutoModel



data = tfds.load('wikipedia')
data = data['train']
data = data.map(lambda x: x['text'], num_parallel_calls = tf.data.AUTOTUNE)

# data = data.map(lambda x: x.decode('utf-8'), num_parallel_calls = tf.data.AUTOTUNE)
data.batch(1024).prefetch(tf.data.AUTOTUNE)


stri = ''
i = 0
for elem in data:
    if i % 2 == 0: stri += elem.numpy().decode()
    else: continue


# with open('string.txt', 'r') as file:
#     string = file.read().replace('\n', '')

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-tiny-5-finetuned-squadv2",
    tokenizer="mrm8488/bert-tiny-5-finetuned-squadv2"
)
with tf.device('/GPU:0'):
    bert_answer = qa_pipeline({
        'context': stri,
        'question': "Tell me something important"
    })

print(bert_answer["answer"])

set_seed(42)
generator = pipeline('text-generation', model='gpt2')

with tf.device('/GPU:0'):
    gpt2_answer = generator(bert_answer["answer"], max_length=30, num_return_sequences=5)

print(gpt2_answer)
