import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Concatenate
import os
import numpy as np
from tensorflow.keras.layers import TextVectorization, Input, Embedding, LSTM, Dense, Concatenate, Attention
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout, Input
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# Load the dataset
dataframe = pd.read_csv('ai_corpus_cleaned.csv')
dataframe = dataframe.dropna()

# Convert texts to lowercase and add start and end tokens to the target sequences
aave_texts = dataframe['AAVE'].str.lower().tolist()
sae_texts = ['<start> ' + text + ' <end>' for text in dataframe['SAE'].str.lower().tolist()]


# Tokenize the texts
def tokenize(lang):
    lang_tokenizer = Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer

input_tensor, inp_lang_tokenizer = tokenize(aave_texts)
target_tensor, targ_lang_tokenizer = tokenize(sae_texts)

# Determine the maximum sequence length for proper model configuration
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

# Split the data into training and validation sets
input_train, input_val, target_train, target_val = train_test_split(input_tensor, target_tensor, test_size=0.2, random_state=21)

# Create a TensorFlow dataset for the training data
BUFFER_SIZE = len(input_train)
# BATCH_SIZE = 64
# dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train)).shuffle(BUFFER_SIZE)
# dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

BATCH_SIZE = 32  # Reduced from a larger size
dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

val_dataset = tf.data.Dataset.from_tensor_slices((input_val, target_val)).batch(BATCH_SIZE, drop_remainder=True)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.bilstm = Bidirectional(LSTM(enc_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform'))
    
    def call(self, x):
        x = self.embedding(x)
        output, forward_h, forward_c, backward_h, backward_c = self.bilstm(x)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        return output, state_h, state_c

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2*self.enc_units))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, num_heads):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(dec_units * 2, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = Dense(vocab_size)
        # MultiHeadAttention layer
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=dec_units)
    
    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        
        # Prepare query, key, value for MultiHeadAttention
        query = self.embedding(x)  # Use decoder input x as query
        key = value = enc_output  # Use encoder output for both key and value
        
        # Multi-head attention
        attention_output, attention_weights = self.multi_head_attention(query=query, value=value, key=key, return_attention_scores=True)
        
        # LSTM output
        lstm_output, state_h, state_c = self.lstm(attention_output)
        
        # Reshape lstm_output to pass through the Dense layer
        output = tf.reshape(lstm_output, (-1, lstm_output.shape[2]))
        
        # Final output
        x = self.fc(output)
        
        return x, state_h, state_c, attention_weights

class Seq2SeqModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size, start_token_index, end_token_index, num_heads):
        super(Seq2SeqModel, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, units, batch_size)
        # Pass num_heads to the decoder
        self.decoder = Decoder(vocab_size, embedding_dim, units, batch_size, num_heads)
        self.batch_size = batch_size
        self.start_token_index = start_token_index
        self.end_token_index = end_token_index


    def call(self, inp, targ, training=True):
        enc_output, enc_hidden_h, enc_hidden_c = self.encoder(inp)
        dec_hidden = enc_hidden_h
        dec_input = tf.expand_dims([self.start_token_index] * self.batch_size, 1)

        # For storing the predictions
        predictions = []

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # Passing enc_output to the decoder
            pred, dec_hidden, _, _ = self.decoder(dec_input, dec_hidden, enc_output)
            predictions.append(pred)

            # Using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1) if training else tf.argmax(pred, axis=1)

        # Concatenate predictions along the time dimension
        predictions = tf.stack(predictions, 1)

        return predictions


vocab_inp_size = len(inp_lang_tokenizer.word_index) + 1  # +1 for padding token
vocab_tar_size = len(targ_lang_tokenizer.word_index) + 1  # +1 for padding token
embedding_dim = 128  # Assuming you're continuing with this embedding dimension
units = 10         # Assuming you're continuing with these units for LSTM
BATCH_SIZE = 32      # Assuming you're continuing with this batch size


# Define the size of the vocabularies
vocab_inp_size = len(inp_lang_tokenizer.word_index) + 1
vocab_tar_size = len(targ_lang_tokenizer.word_index) + 1

# Find start and end token indices for the target sequences
start_token_index = targ_lang_tokenizer.word_index['<start>']
end_token_index = targ_lang_tokenizer.word_index['<end>']

num_heads = 4  # Example value, adjust based on your model's needs
seq2seq_model = Seq2SeqModel(vocab_size=vocab_inp_size,
                             embedding_dim=embedding_dim,
                             units=units,
                             batch_size=BATCH_SIZE,
                             start_token_index=start_token_index,
                             end_token_index=end_token_index,
                             num_heads=num_heads)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)


from tensorflow.keras.callbacks import EarlyStopping

# Early Stopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


# Assuming `Seq2SeqModel` is initialized as `seq2seq_model`
seq2seq_model.compile(optimizer=optimizer, loss=loss_function)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden_h, enc_hidden_c = seq2seq_model.encoder(inp)
        dec_hidden = enc_hidden_h
        dec_input = tf.expand_dims([seq2seq_model.start_token_index] * seq2seq_model.batch_size, 1)
        
        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _, _ = seq2seq_model.decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)
        
    batch_loss = (loss / int(targ.shape[1]))
    variables = seq2seq_model.encoder.trainable_variables + seq2seq_model.decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss


print('Before training')
# Initialize the hidden state before starting the training loop
enc_hidden = seq2seq_model.encoder.initialize_hidden_state()
steps_per_epoch = len(input_train) // BATCH_SIZE

EPOCHS = 100  # Define the number of epochs you want to train for

for epoch in range(EPOCHS):
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss
        
        # Optional: Print batch loss
        if batch % 100 == 0:
            print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy()}')
            
    # Print total loss after each epoch
    print(f'Epoch {epoch+1} Loss {total_loss / steps_per_epoch:.4f}')
    # Re-initialize the hidden state at the start of each epoch
    enc_hidden = seq2seq_model.encoder.initialize_hidden_state()

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import tensorflow as tf

def translate(model, sentence, inp_lang, targ_lang, max_length_inp, max_length_targ):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ') if i in inp_lang.word_index]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    # Adjusted to access the encoder's units. Change `enc_units` as per your model's attribute
    enc_hidden = [tf.zeros((1, model.encoder.enc_units))]
    enc_out, enc_hidden_h, enc_hidden_c = model.encoder(inputs)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        # Correctly unpacking the expected four return values from the decoder's call method
        predictions, dec_hidden, _, attention_weights = model.decoder(dec_input, dec_hidden, enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word.get(predicted_id, '<unk>') + ' '

        if targ_lang.index_word.get(predicted_id) == '<end>':
            return result.strip(), sentence

        dec_input = tf.expand_dims([predicted_id], 0)

    return result.strip(), sentence


from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

def evaluate_and_calculate_bleu(model, dataset, inp_lang, targ_lang, max_length_inp, max_length_targ):
    actual, predicted = [], []
    smoothing_function = SmoothingFunction().method1  # Choose a smoothing function as necessary
    
    for input_tensor, target_tensor in dataset:
        target_sentence = ' '.join([targ_lang.index_word[i] for i in target_tensor.numpy()[0] if i > 0])
        input_sentence = ' '.join([inp_lang.index_word[i] for i in input_tensor.numpy()[0] if i > 0])
        translation, _ = translate(model, input_sentence, inp_lang, targ_lang, max_length_inp, max_length_targ)
        
        actual.append([target_sentence.split()])
        predicted.append(translation.split())
    
    # Calculate BLEU score
    bleu_score = corpus_bleu(actual, predicted, smoothing_function=smoothing_function)
    return bleu_score

# Assuming `input_val` and `target_val` are your validation sets, and you've defined inp_lang_tokenizer and targ_lang_tokenizer
val_dataset = tf.data.Dataset.from_tensor_slices((input_val, target_val)).batch(1)

# Calculate BLEU score for the validation set
bleu_score = evaluate_and_calculate_bleu(seq2seq_model, val_dataset, inp_lang_tokenizer, targ_lang_tokenizer, max_length_inp, max_length_targ)
print(f'BLEU score on validation set: {bleu_score:.4f}')
