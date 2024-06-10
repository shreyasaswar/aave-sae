import itertools
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Concatenate, Attention
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

print("Experiment with BPE, start end tokens")

# Load dataset
dataframe = pd.read_csv('pro_corpus.csv').dropna()
dataframe['AAVE'] = '[START] ' + dataframe['AAVE'].str.lower() + ' [END]'
dataframe['SAE'] = '[START] ' + dataframe['SAE'].str.lower() + ' [END]'

# Split the data
aave_train, aave_test, sae_train, sae_test = train_test_split(dataframe['AAVE'], dataframe['SAE'], test_size=0.2)

# Train a BPE Tokenizer (Hugging Face Tokenizers)
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=['[UNK]', '[START]', '[END]'])
tokenizer.train_from_iterator(list(aave_train) + list(sae_train), trainer=trainer)

# Tokenize and Pad Sequences Function
def tokenize_and_pad(texts, max_length=50):
    tokenized_texts = [tokenizer.encode(text).ids for text in texts]
    padded_texts = [ids[:max_length] + [0] * (max_length - len(ids)) for ids in tokenized_texts]
    return np.array(padded_texts)

# Preparing the datasets
max_length = 30
aave_train_tok = tokenize_and_pad(aave_train, max_length)
sae_train_tok = tokenize_and_pad(sae_train, max_length)
aave_test_tok = tokenize_and_pad(aave_test, max_length)
sae_test_tok = tokenize_and_pad(sae_test, max_length)

# Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(((aave_train_tok, sae_train_tok[:, :-1]), sae_train_tok[:, 1:])).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices(((aave_test_tok, sae_test_tok[:, :-1]), sae_test_tok[:, 1:])).batch(64)


# In[30]:


patience_range = [3, 5]
units_range = [128, 256]
learning_rate_range = [0.001,0.0001, 0.01]
activation_functions = ['relu', 'tanh']
dropout_rates = [0.3, 0.5, 0.7]
regularizers = [None, 'l2']
optimizers = ['adam']#,'rmsprop', 'sgd']
embedding_dim_range = [128, 256]

# Generate all combinations of hyperparameters
hyperparameter_combinations = list(itertools.product(
    patience_range, units_range, learning_rate_range, 
    dropout_rates, embedding_dim_range, optimizers, activation_functions, regularizers))


# In[31]:


def build_model(vocab_size, embedding_dim=256, units=128, dropout=0.5, regularizer=None, learning_rate=0.001, optimizer_choice='adam'):
    # Encoder
    encoder_input = Input(shape=(None,), dtype='int32', name='encoder_input')
    encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_input)
    reg = regularizers.l2(0.001) if regularizer == 'l2' else None
    encoder_bilstm = Bidirectional(LSTM(units, return_sequences=True, return_state=True, dropout=dropout, recurrent_dropout=dropout, kernel_regularizer=regularizers.l2(0.001) if regularizer == 'l2' else None))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(encoder_embedding)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,), dtype='int32', name='decoder_input')
    decoder_embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    decoder_lstm = LSTM(units * 2, return_sequences=True, return_state=True, dropout=dropout,recurrent_dropout=dropout, kernel_regularizer=regularizers.l2(0.001) if regularizer == 'l2' else None)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # Attention layer
    attention_layer = Attention(use_scale=True)
    attention_output = attention_layer([decoder_outputs, encoder_outputs])
    decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attention_output])

    # Dense layer for prediction
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Define the model
    model = Model([encoder_input, decoder_inputs], decoder_outputs)

    # Optimizer selection based on the directly imported classes
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    
    
    # Compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model



# In[32]:


# Function to save results to CSV
def append_results_to_csv(results, filename="modelMH7_hyperparameters_results.csv"):
    pd.DataFrame([results]).to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)


# In[33]:


def calculate_bleu_score(model, dataset, tokenizer):
    bleu_scores = []
    smoothie = SmoothingFunction().method4
    for input_seq, target_seq in dataset.unbatch().batch(1):
        target_seq = target_seq.numpy()[0]
        prediction = model.predict(input_seq)
        pred_tokens = np.argmax(prediction, axis=-1)[0]
        pred_sentence = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        actual_sentence = tokenizer.decode(target_seq[1:], skip_special_tokens=True) # Skip [START] token
        bleu_score = sentence_bleu([actual_sentence.split()], pred_sentence.split(), smoothing_function=smoothie)
        bleu_scores.append(bleu_score)
    average_bleu_score = np.mean(bleu_scores)
    return average_bleu_score


# In[34]:


hyperparameter_combinations


# In[ ]:


# Main loop
for combination in hyperparameter_combinations:
    patience, units, learning_rate, dropout, embedding_dim, optimizer_choice, activation_function, regularizer = combination
    
    # Assuming tokenizer is prepared and vocab_size is available
    vocab_size = tokenizer.get_vocab_size()
    model = build_model(vocab_size=vocab_size, embedding_dim=embedding_dim, units=units,
                        dropout=dropout, learning_rate=learning_rate, optimizer_choice=optimizer_choice)
    
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience, verbose=2)
    
    # Fit the model
    history = model.fit(train_dataset, epochs=100, validation_data=test_dataset,
                        callbacks=[early_stopping_callback], verbose=2)
    
    bleu_score = calculate_bleu_score(model, test_dataset, tokenizer)
    
    # Collect and save results
    results = {
        'patience': patience, 'units': units, 'learning_rate': learning_rate,
        'dropout': dropout, 'embedding_dim': embedding_dim, 'optimizer': optimizer_choice,
        'activation_function':activation_function,'regularizer':regularizer,
        'bleu_score': bleu_score, 'epochs': len(history.history['loss']),
        'final_train_loss': history.history['loss'][-1], 'final_val_loss': history.history['val_loss'][-1]
    }
    append_results_to_csv(results)
    print(results)

print("Hyperparameter tuning completed.")



