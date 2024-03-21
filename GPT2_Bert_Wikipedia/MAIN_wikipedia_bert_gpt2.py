import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import pipeline, set_seed


data = tfds.load('wikipedia')
data = data['train']
data = data.map(lambda x: x['text'], num_parallel_calls = tf.data.AUTOTUNE)
data = data.shuffle(512).batch(128).prefetch(tf.data.AUTOTUNE)
data_as_tensors = []
data_again = []

for elem in data.take(20):
    data_as_tensors.append(elem)

encoding = 'ISO-8859-1'
string =""

for elem in data_as_tensors:
    string += elem.numpy()[0].decode(encoding)

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-tiny-5-finetuned-squadv2",
    tokenizer="mrm8488/bert-tiny-5-finetuned-squadv2"
)

bert_answer = qa_pipeline({
    'context': string,
    'question': "Tell me something important"

})

print(bert_answer["answer"])

generator = pipeline('text-generation', model='gpt2')
set_seed(42)
gpt2_answer = generator(bert_answer["answer"], max_length=30, num_return_sequences=5)
print(gpt2_answer)
