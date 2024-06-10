#model 2


This enhanced translation model is designed to translate AAVE to SAE. It uses a more complex encoder-decoder architecture with two LSTM layers and an improved attention mechanism.

How to Use
Dependencies:
Ensure you have the necessary libraries installed:
TensorFlow
Pandas
Matplotlib
nltk


Data Preparation:
Ensure pro_corpus.csv is in the project directory.
The data should contain two columns: "AAVE" and "SAE".


Model Training:
Run the code to:
Load and preprocess the data.
Split the data into training and test sets.
Train the translation model using two-layer LSTM architecture with callbacks.
The best model is saved as best_model_M2.h5.


Evaluation:
The code calculates BLEU scores for translations on the test dataset.
Results are saved to translation_results_M2.csv.
Training and validation loss curves are plotted and saved as loss_curve_M2.png.