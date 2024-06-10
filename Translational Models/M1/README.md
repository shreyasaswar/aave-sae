#model 1


This model is designed to translate African American Vernacular English (AAVE) to Standard American English (SAE). It uses an encoder-decoder framework with a single-layer LSTM and an attention mechanism.

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
Train the translation model with callbacks (early stopping and checkpoint).
The best model is saved as best_model_M1.h5.

Evaluation:
The code calculates BLEU scores for translations on the test dataset.
Results are saved to translation_results_M1.csv.
Training and validation loss curves are plotted and saved as loss_curve_M1.png.