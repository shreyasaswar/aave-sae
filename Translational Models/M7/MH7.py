#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization, Input, Embedding, LSTM, Dense, Concatenate, Attention
from tensorflow.keras.models import Model
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[6]:


# Assuming the CSV has been read into `dataframe`
dataframe = pd.read_csv('pro_corpus.csv')
aave_texts = dataframe['AAVE'].str.lower().tolist()
sae_texts = ["[start] " + text + " [end]" for text in dataframe['SAE'].str.lower().tolist()]

aave_train, aave_test, sae_train, sae_test = train_test_split(
    aave_texts, sae_texts, test_size=0.2, random_state=21)


# In[7]:


max_vocab_size = 20000
sequence_length = 30

aave_vectorization = tf.keras.layers.TextVectorization(max_tokens=max_vocab_size, output_sequence_length=sequence_length)
sae_vectorization = tf.keras.layers.TextVectorization(max_tokens=max_vocab_size, output_sequence_length=sequence_length + 1)  # +1 for [start]/[end] tokens

aave_vectorization.adapt(aave_train)
sae_vectorization.adapt(sae_train)


# In[8]:


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)


# In[9]:


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


# In[10]:


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


# In[11]:


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


# In[12]:


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        
        # Adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


# In[13]:


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


# In[14]:


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(pe_target, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # Adding embedding and positional encoding.
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i, dec_layer in enumerate(self.dec_layers):
            x, block1, block2 = dec_layer(x, enc_output, training, look_ahead_mask, padding_mask)

            # Store attention weights, could be useful for visualization or analysis
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


# In[23]:


import tensorflow as tf
import numpy as np

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention_logits = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights

# Skipping other necessary components (e.g., EncoderLayer, DecoderLayer, Encoder, Decoder) for brevity

# class Transformer(tf.keras.Model):
#     def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
#         super(Transformer, self).__init__()
#         self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
#         self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

#         self.final_layer = tf.keras.layers.Dense(target_vocab_size)

#     def call(self, inputs, training):
#         inp, tar = inputs['inputs'], inputs['dec_inputs']

#         enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar)

#         enc_output = self.encoder(inp, training, enc_padding_mask)
#         dec_output, attention_weights = self.decoder(tar, enc_output, training, combined_mask, dec_padding_mask)

#         final_output = self.final_layer(dec_output)

#         return final_output

#     def create_masks(self, inp, tar):
#         enc_padding_mask = create_padding_mask(inp)
#         dec_padding_mask = create_padding_mask(inp)

#         look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
#         dec_target_padding_mask = create_padding_mask(tar)
#         combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

#         return enc_padding_mask, combined_mask, dec_padding_mask

# # Define other necessary components (e.g., Encoder, Decoder) and training process as needed

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        # Split inputs
        inp, tar = inputs['inputs'], inputs['dec_inputs']
        
        # Call the create_masks function here
        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar)

        # Encoder output
        enc_output = self.encoder(inp, training, enc_padding_mask)
        
        # Decoder output
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, combined_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)
        return final_output

    def create_masks(self, inp, tar):
        # Create padding mask for input
        enc_padding_mask = create_padding_mask(inp)
        
        # Create look-ahead mask and padding mask for target
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        # Decoder padding mask for second attention block in decoder
        dec_padding_mask = create_padding_mask(inp)
        
        return enc_padding_mask, combined_mask, dec_padding_mask



# In[24]:


# Assuming aave_vectorization and sae_vectorization have been adapted on the respective datasets
def make_dataset(aave, sae):
    aave_ds = aave_vectorization(aave)
    sae_ds = sae_vectorization(sae)
    # Decoder inputs use the [:, :-1] slices of sae_ds, and the targets are the [:, 1:] slices
    input_ds = {"inputs": aave_ds, "dec_inputs": sae_ds[:, :-1]}
    target_ds = sae_ds[:, 1:]  # Targets are offset by 1 to predict the next token
    return tf.data.Dataset.from_tensor_slices((input_ds, target_ds)).batch(64).cache().prefetch(tf.data.experimental.AUTOTUNE)

train_ds = make_dataset(aave_train, sae_train)
val_ds = make_dataset(aave_test, sae_test)


# In[25]:


# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, d_model, warmup_steps=4000):
#         super(CustomSchedule, self).__init__()
#         self.d_model = tf.cast(d_model, tf.float32)
#         self.warmup_steps = warmup_steps

#     def __call__(self, step):
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps ** -1.5)
#         arg3 = tf.math.rsqrt(self.d_model)
#         return arg3 * tf.math.minimum(arg1, arg2)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)  # Cast step to float32
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        arg3 = tf.math.rsqrt(self.d_model)
        return arg3 * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model=512)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)



def accuracy(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

# Hyperparameters for the Transformer
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
input_vocab_size = aave_vectorization.vocabulary_size() + 2  # +2 for start/end tokens
target_vocab_size = sae_vectorization.vocabulary_size() + 2  # +2 for start/end tokens
pe_input = max([len(sentence.split()) for sentence in aave_texts])  # or a fixed number like 1000
pe_target = max([len(sentence.split()) for sentence in sae_texts])  # or a fixed number like 1000
dropout_rate = 0.1

# Instantiate the Transformer model
transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                          input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size,
                          pe_input=pe_input, pe_target=pe_target, rate=dropout_rate)


# In[26]:


transformer.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])


# In[27]:


EPOCHS = 50

# Define the checkpoint path and the checkpoint manager.
# This saves checkpoints to disk.
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# If a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])


# In[40]:


def translate(sentence, transformer, inp_lang_vectorizer, tar_lang_vectorizer):
    # Preprocess and tokenize the input sentence
    sentence = tf.convert_to_tensor([sentence])
    sentence = inp_lang_vectorizer(sentence)

    encoder_input = sentence

    # Assuming start token is 2 and end token is 3 for illustration; adjust according to your setup
    start, end = 2, 3
    output = tf.convert_to_tensor([start])
    output = tf.expand_dims(output, 0)

    for i in range(40):  # Assuming a maximum length of 40 tokens
        predictions = transformer({'inputs': encoder_input, 'dec_inputs': output}, training=False)

        # Select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # Concatenate the predicted_id to the output
        output = tf.concat([output, predicted_id], axis=-1)

        # Check if the last predicted word is equal to the end token
        if predicted_id.numpy()[0][0] == end:
            break

    output = tf.squeeze(output, axis=0).numpy()

    # Convert sequence of IDs to text
    vocabulary = tar_lang_vectorizer.get_vocabulary()
    predicted_sentence = ' '.join([vocabulary[i] for i in output if i < len(vocabulary) and i != start])

    # Optionally, strip the part after the end token if it was included
    end_token_index = vocabulary.index('[end]') if '[end]' in vocabulary else -1
    if end_token_index != -1:
        predicted_sentence = predicted_sentence.split('[end]')[0]

    return predicted_sentence.strip()


# In[41]:


from nltk.translate.bleu_score import corpus_bleu

def calculate_bleu_score(input_texts, target_texts, transformer, inp_lang_vectorizer, tar_lang_vectorizer):
    references = []
    candidates = []

    for i, sentence in enumerate(input_texts):
        target = target_texts[i][len("[start] "):-len(" [end]")]  # Remove the start and end tokens from the target
        translation = translate(sentence, transformer, inp_lang_vectorizer, tar_lang_vectorizer)
        
        references.append([target.split(' ')])  # BLEU references need to be tokenized and wrapped in a list
        candidates.append(translation.split(' '))

    bleu_score = corpus_bleu(references, candidates)
    return bleu_score




# Example usage
bleu_score = calculate_bleu_score(aave_test[:20], sae_test[:20], transformer, aave_vectorization, sae_vectorization)  # Use a slice for quick testing
print(f"BLEU score on test set: {bleu_score}")






