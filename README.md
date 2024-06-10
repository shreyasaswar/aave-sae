# aave-sae
Research work on the AAVE project under Prof. Sunandan Chakraborty


This repository contains implementations of various translation models designed to convert African American Vernacular English (AAVE) to Standard American English (SAE). Each model builds upon the previous one, incorporating architectural changes and hyperparameter tuning to improve translation performance.

Model 1: Basic Model
Design:

Encoder-Decoder: Uses a single-layer LSTM for both the encoder and decoder.
Attention: Incorporates a single attention layer for better alignment of sequences.
Text Vectorization: Pads sequences to a length of 100.
Evaluation: Implements BLEU score calculation with training and validation loss curves.
Reasoning:
The initial model provides a baseline for translation performance with a straightforward architecture. The attention mechanism helps the decoder focus on relevant words during translation, while BLEU scores and loss curves offer insights into translation quality.

Model 2: Enhanced Model
Design:

Encoder-Decoder: Adds a second LSTM layer to both the encoder and decoder.
Attention: Uses a shorter sequence length (30) for vectorization because the more than 90 percent of sentences in the corpus are below 30 token length.
Batch Prefetching: Utilizes tf.data.AUTOTUNE to optimize the data pipeline.
Early Stopping: Patience set to 10 epochs.
Evaluation: Saves results in separate BLEU score files.
Reasoning:
The deeper LSTM architecture improves context capture, and shorter sequence lengths reduce padding and better captures the nuances. Prefetching improves data loading efficiency, and increased early stopping patience helps prevent premature termination.

Model 3: New Model
Design:

Encoder: Utilizes a bidirectional LSTM to capture context from both directions.
Decoder: Single-layer LSTM with multi-head attention.
Teacher Forcing: Ensures proper sequence generation during training.
Loss Function: Custom loss function handles padding efficiently.
Evaluation: Uses BLEU score calculation via the NLTK library.
Reasoning:
The bidirectional encoder and multi-head attention layer allow the decoder to focus on relevant input segments simultaneously, improving translation. Teacher forcing reduces model drift, and the custom loss function manages padding better.

Model 4: Latest Model
Design:

Encoder: Bidirectional LSTM with concatenated hidden states.
Decoder: Single-layer LSTM with a single attention mechanism.
Hyperparameter Tuning: Optimized through combinations of dropout, activation functions, and regularization.
Evaluation: Saves BLEU scores and loss curves based on varying hyperparameter combinations.
Reasoning:
Hyperparameter tuning experiments aim to identify the best-performing architecture. Bidirectional LSTMs enhance context capture, while carefully chosen combinations of dropout and activation improve the model's robustness. Each experiment is saved for future reference, making it easier to compare the performance across combinations.

Model 5: Fifth Model
Design:

Encoder: Bidirectional LSTM with concatenated hidden states and start/end tokens.
Decoder: Single-layer LSTM with a single attention mechanism.
Hyperparameter Tuning: Optimized through combinations of activation functions and optimizers.
Evaluation: Uses BLEU score calculation with start/end tokens omitted.
Reasoning:
The inclusion of start/end tokens helps clarify the sequence boundaries during translation. The bidirectional encoder enhances input understanding, while combinations of optimizers, activation functions, and regularization aim to improve the model's robustness and generalization. BLEU score evaluation excludes these tokens for accurate assessment.

Model 6: Sixth Model
Design:

Encoder: Bidirectional LSTM with concatenated hidden states and BPE tokenization.
Decoder: Single-layer LSTM with a single attention mechanism.
Hyperparameter Tuning: Combines various activation functions, optimizers, and dropout rates.
Evaluation: Uses BLEU score calculation via the NLTK library.
Reasoning:
The use of byte-pair encoding (BPE) increases vocabulary flexibility, and bidirectional LSTMs improve context understanding. The comprehensive tuning of hyperparameters seeks to find optimal settings that balance training performance and generalization. BLEU score calculation provides an accurate measure of translation quality.

Model 7: Seventh Model
Design:

Encoder-Decoder: Utilizes a Transformer architecture with multi-head attention and position encoding.
Hyperparameter Settings: Incorporates fixed layers and units for consistency.
Evaluation: Uses BLEU score calculation via the NLTK library.
Reasoning:
The Transformer architecture improves context capture using multi-head attention across multiple layers. Positional encoding helps maintain word ordering in sequence translations, and a consistent configuration allows comparison across different datasets. BLEU score evaluation provides an accurate measure of translation performance.

Model 8: Eighth Model
Design:

Encoder: Bidirectional LSTM with contextualized embeddings.
Decoder: Single-layer LSTM with a single attention mechanism.
Hyperparameter Tuning: Comprehensive combinations for best results.
Evaluation: Uses BLEU score calculation via NLTK and was submitted to a paper at ACM Compass.
Reasoning:
The model leverages contextualized word embeddings to better capture input meaning, while attention and bidirectional LSTMs focus on sequence boundaries. This model was refined through hyperparameter combinations and achieved the best results for the ACM Compass paper submission.

Model 9: Ninth Model
Design:

Encoder: Bidirectional LSTM with BPE tokenization.
Decoder: Single-layer LSTM with a single attention mechanism.
Hyperparameter Tuning: Comprehensive combinations were explored for best results.
Evaluation: Uses BLEU score calculation via NLTK.
Reasoning:
The BPE tokenization provides more robust vocabulary control, while bidirectional LSTMs and a single attention mechanism ensure strong contextual understanding. This model benefits from extensive hyperparameter tuning and padding mechanisms, offering efficient training with accurate BLEU score evaluation.

