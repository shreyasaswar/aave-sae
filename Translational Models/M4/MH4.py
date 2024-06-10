#!/usr/bin/env python
# coding: utf-8


import itertools
import os
import tensorflow as tf
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





dataframe = pd.read_csv('/N/u/saswar/Carbonate/AAVE/ai_corpus_cleaned.csv')
dataframe = dataframe.dropna()

assert 'AAVE' in dataframe.columns and 'SAE' in dataframe.columns

# Preparing the dataset
aave_texts = dataframe['AAVE'].str.lower().tolist()
sae_texts = dataframe['SAE'].str.lower().tolist()

# Split the data into train and test sets
aave_train, aave_test, sae_train, sae_test = train_test_split(
    aave_texts, sae_texts, test_size=0.2, random_state=21)

# Convert the train and test data into TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices({
    'aave': aave_train,
    'sae': sae_train
})
test_dataset = tf.data.Dataset.from_tensor_slices({
    'aave': aave_test,
    'sae': sae_test
})





BUFFER_SIZE = len(aave_train)  # Use the size of the train dataset

# Adjust batch sizes
train_batch_size = 16
test_batch_size = 4

# Shuffle and batch the train dataset
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(train_batch_size, drop_remainder=True)

# Batch the test dataset
test_dataset = test_dataset.batch(test_batch_size, drop_remainder=True)

# Text Vectorization
aave_vectorization = TextVectorization(output_mode='int', output_sequence_length=30)
sae_vectorization = TextVectorization(output_mode='int', output_sequence_length=30)

aave_texts = train_dataset.map(lambda x: x['aave'])
sae_texts = train_dataset.map(lambda x: x['sae'])

aave_vectorization.adapt(aave_texts)
sae_vectorization.adapt(sae_texts)

aave_vocab_size = len(aave_vectorization.get_vocabulary())
sae_vocab_size = len(sae_vectorization.get_vocabulary())


# In[9]:


# Function to transform the dataset
def split_input_target(batch):
    input_text = batch['aave']
    target_text = batch['sae']
    input_data = aave_vectorization(input_text)
    target_data = sae_vectorization(target_text)
    return {'encoder_input': input_data, 'decoder_input': target_data[:, :-1]}, target_data[:, 1:]

# Apply the function to each item in the dataset
train_dataset = train_dataset.map(split_input_target)
test_dataset = test_dataset.map(split_input_target)


# In[10]:


# Define the ranges for each hyperparameter
patience_range = [8, 10]
units_range = [128,256]
learning_rate_range = [0.001, 0.01]
activation_functions = ['relu', 'tanh']
dropout_rates = [0.3, 0.5, 0.7]
regularizers = [None, 'l2']
optimizers = ['adam']#, 'sgd', 'rmsprop']
embedding_dim_range = [128, 256]  # Added embedding dimension range
# Add more hyperparameters as needed


# In[11]:


# Create all combinations of hyperparameters
hyperparameter_combinations = list(itertools.product(
    patience_range, units_range, learning_rate_range, activation_functions, dropout_rates, regularizers, optimizers, embedding_dim_range))

from tensorflow.keras.layers import Bidirectional, Concatenate

def build_model(units, activation, dropout, regularizer, learning_rate, optimizer, embedding_dim):
    # Encoder
    encoder_input = Input(shape=(None,), dtype='int64', name='encoder_input')
    encoder_embedding = Embedding(input_dim=aave_vocab_size, output_dim=embedding_dim)(encoder_input)
    encoder_bilstm = Bidirectional(LSTM(units, return_sequences=True, return_state=True, dropout=dropout, recurrent_dropout=dropout, kernel_regularizer=regularizer if regularizer == 'l2' else None))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(encoder_embedding)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_state = [state_h, state_c]

    # Decoder
    decoder_input = Input(shape=(None,), dtype='int64', name='decoder_input')
    decoder_embedding = Embedding(input_dim=sae_vocab_size, output_dim=embedding_dim)(decoder_input)
    decoder_lstm = LSTM(units * 2, return_sequences=True, return_state=True, dropout=dropout, recurrent_dropout=dropout, kernel_regularizer=regularizer if regularizer == 'l2' else None)  # Double the units because of BiLSTM
    decoder_lstm_output, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_state)

    # Attention Mechanism
    attention_layer = Attention(use_scale=True)
    attention_output = attention_layer([decoder_lstm_output, encoder_outputs])

    # Concatenation
    decoder_concat_input = Concatenate(axis=-1)([decoder_lstm_output, attention_output])

    # Output Layer
    decoder_dense = Dense(sae_vocab_size, activation='softmax')(decoder_concat_input)

    model = Model([encoder_input, decoder_input], decoder_dense)

    # Compile the model
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def calculate_bleu_score(model, dataset):
    bleu_scores = []
    smoothie = SmoothingFunction().method4 

    for batch in dataset:
        input_data, target_data = batch
        predictions = np.argmax(model.predict(input_data), axis=-1)

        for input_seq, pred, actual in zip(input_data['encoder_input'], predictions, target_data):
            input_sentence = [aave_vectorization.get_vocabulary()[i] for i in input_seq.numpy() if i != 0]
            pred_sentence = [sae_vectorization.get_vocabulary()[i] for i in pred if i != 0]
            actual_sentence = [sae_vectorization.get_vocabulary()[i] for i in actual.numpy() if i != 0]

            if len(pred_sentence) == 0 or len(actual_sentence) == 0:
                continue

            bleu_score = sentence_bleu([actual_sentence], pred_sentence, smoothing_function=smoothie)
            bleu_scores.append(bleu_score)

    average_bleu_score = np.mean(bleu_scores) if len(bleu_scores) > 0 else 0.0
    return average_bleu_score



def train_and_evaluate_model(patience, units, learning_rate, activation, dropout, regularizer, optimizer,embedding_dim):
    # Build the model with the specified hyperparameters
    model = build_model(units, activation, dropout, regularizer, learning_rate, optimizer,embedding_dim)

    # Early Stopping Callback
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=2)

    # Train the model with a fixed number of epochs (100) but use early stopping
    history = model.fit(train_dataset, epochs=100, validation_data=test_dataset, callbacks=[early_stopping_callback],verbose=2)

    # The epoch at which training stopped
    stopped_epoch = early_stopping_callback.stopped_epoch

    # Evaluate the model (e.g., calculate BLEU score)
    bleu_score = calculate_bleu_score(model, test_dataset)
    
    return bleu_score, model, stopped_epoch, history


# In[ ]:


results = []

# Iterate over all combinations
for combination in hyperparameter_combinations:
    patience, units, learning_rate, activation, dropout, regularizer, optimizer, embedding_dim = combination
    bleu_score, trained_model, stopped_epoch, history = train_and_evaluate_model(patience, units, learning_rate, activation, dropout, regularizer, optimizer, embedding_dim)

    # Print the results of the current experiment
    print(f"Experiment with patience={patience}, units={units}, learning_rate={learning_rate}, "
          f"activation={activation}, dropout={dropout}, regularizer={regularizer}, optimizer={optimizer}, "
          f"embedding_dim={embedding_dim} completed.")
    print(f"Stopped epoch: {stopped_epoch}, BLEU score: {bleu_score:.4f}\n")
    
    # Save the model
    model_dir_name = f'modelM3_ai_patience{patience}_units{units}_lr{learning_rate}_act{activation}_dropout{dropout}_reg{regularizer}_opt{optimizer}_embed{embedding_dim}'
    os.makedirs(model_dir_name, exist_ok=True)
    model_path = os.path.join(model_dir_name, 'model.h5')
    trained_model.save(model_path)
    
    # Save the loss curve
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    loss_curve_path = os.path.join(model_dir_name, 'loss_curve.png')
    plt.savefig(loss_curve_path)
    plt.close()
    
    # Store results
    results.append({
        'patience': patience, 'units': units, 'learning_rate': learning_rate, 'activation': activation, 
        'dropout': dropout, 'regularizer': regularizer, 'optimizer': optimizer, 'embedding_dim': embedding_dim, 'stopped_epoch': stopped_epoch,
        'bleu_score': bleu_score
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('hyperparameter_tuning_results_M3_ai_corpus.csv', index=False)

print("Hyperparameter tuning completed. Results saved to 'hyperparameter_tuning_results_M3_ai_corpus.csv'.")





