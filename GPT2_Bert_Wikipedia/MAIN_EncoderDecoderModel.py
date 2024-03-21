import tensorflow as tf
from transformers import TFBertModel, TFGPT2Model, BertTokenizer, GPT2Tokenizer, EncoderDecoderModel

class EncoderDecoder(tf.keras.Model):
    def __init__(self, bert_model, gpt2_model):
        super(EncoderDecoder, self).__init__()
        self.bert_model = bert_model
        self.gpt2_model = gpt2_model

    def call(self, inputs):
        # Encoder (BERT)
        encoder_output = self.bert_model(inputs)[0]  # BERT returns a tuple, we only need the output
        # Decoder (GPT-2)
        decoder_output = self.gpt2_model(inputs)[0]  # GPT-2 also returns a tuple, we only need the output
        return encoder_output, decoder_output


# Load BERT model and tokenizer
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load GPT-2 model and tokenizer
gpt2_model = TFGPT2Model.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.decode()

# initialise the encoder-decoder class
model = EncoderDecoder(bert_model, gpt2_model)

# Initialize the encoder-decoder model
encoder_decoder_model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "gpt2")

# Example input sentence
input_text = "Example input sentence for testing"

# Tokenize input text for both models
input_ids = bert_tokenizer.encode(input_text, add_special_tokens=True, return_tensors='tf')

breakpoint()
# Pass input through the model
outputs = model(input_ids)
breakpoint()
output = encoder_decoder_model(input_ids, decoder_input_ids=input_ids)

# Decode the generated output
decoded_output = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated output
print("Generated Output:", decoded_output)