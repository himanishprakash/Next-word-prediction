import gradio as gr
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import numpy as np
import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Paths to the uploaded models
repo_id = "himanishprak23/lstm_rnn"
lstm_filename = "model_lstm_4.keras"  
rnn_filename = "model_rnn_1.keras"
lstm_model_path = hf_hub_download(repo_id=repo_id, filename=lstm_filename)
rnn_model_path = hf_hub_download(repo_id=repo_id, filename=rnn_filename)

# Specify the repository and the CSV file name
# Specify the repository and the CSV file name
repo_path = "himanishprak23/commentry_Data"
file_name = "df_commentary_new.csv"

# Load the dataset
dataset = load_dataset(repo_path, data_files=file_name, split='train')
data_text = dataset.to_pandas()


# Load the LSTM model
lstm_model = load_model(lstm_model_path)

# Load the RNN model
rnn_model = load_model(rnn_model_path)



# Check the embedding layer's input dimension for LSTM
embedding_layer = lstm_model.layers[0]
vocab_size = embedding_layer.input_dim

# Initialize and fit the tokenizer with limited vocabulary size
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(data_text['Modified_Commentary'])

# Define the maximum sequence length (adjust based on your model training)
max_sequence_length = 153

# Define the text generation function for LSTM
def generate_with_lstm(commentary_text, num_words):
    # Tokenize the input text
    input_sequence = tokenizer.texts_to_sequences([commentary_text])
    input_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length)
    
    # Convert to tensor
    input_tensor = tf.convert_to_tensor(input_sequence)
    
    # Generate the next words
    generated_sequence = []
    for _ in range(num_words):
        # Get model predictions
        output = lstm_model.predict(input_tensor)
        
        # Get the index of the most probable next word
        next_word_index = np.argmax(output[0], axis=-1)
        
        # Add the predicted word to the sequence
        generated_sequence.append(next_word_index)
        
        # Append the predicted word to the input sequence
        input_sequence = np.append(input_sequence[0][1:], next_word_index).reshape(1, -1)
        input_tensor = tf.convert_to_tensor(input_sequence)
    
    # Convert indices back to words
    reverse_word_index = {value: key for key, value in tokenizer.word_index.items() if value < vocab_size}
    generated_words = [reverse_word_index.get(i, '') for i in generated_sequence]
    
    # Combine the input text with the generated words
    generated_text = commentary_text + ' ' + ' '.join(generated_words)
    
    return generated_text

# Define the text generation function for RNN
def generate_with_rnn(commentary_text, num_words):
    # Tokenize the input text
    input_sequence = tokenizer.texts_to_sequences([commentary_text])
    input_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length)
    
    # Convert to tensor
    input_tensor = tf.convert_to_tensor(input_sequence)
    
    # Generate the next words
    generated_sequence = []
    for _ in range(num_words):
        # Get model predictions
        output = rnn_model.predict(input_tensor)
        
        # Get the index of the most probable next word
        next_word_index = np.argmax(output[0], axis=-1)
        
        # Add the predicted word to the sequence
        generated_sequence.append(next_word_index)
        
        # Append the predicted word to the input sequence
        input_sequence = np.append(input_sequence[0][1:], next_word_index).reshape(1, -1)
        input_tensor = tf.convert_to_tensor(input_sequence)
    
    # Convert indices back to words
    reverse_word_index = {value: key for key, value in tokenizer.word_index.items() if value < vocab_size}
    generated_words = [reverse_word_index.get(i, '') for i in generated_sequence]
    
    # Combine the input text with the generated words
    generated_text = commentary_text + ' ' + ' '.join(generated_words)
    
    return generated_text

# Load GPT-2 models and tokenizers
trained_tokenizer = GPT2Tokenizer.from_pretrained("Kumarkishalaya/GPT-2-next-word-prediction")
trained_model = GPT2LMHeadModel.from_pretrained("Kumarkishalaya/GPT-2-next-word-prediction")
untrained_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
untrained_model = GPT2LMHeadModel.from_pretrained("gpt2")

device = "cuda" if torch.cuda.is_available() else "cpu"
trained_model.to(device)
untrained_model.to(device)

# Set pad_token to eos_token
trained_tokenizer.pad_token = trained_tokenizer.eos_token
untrained_tokenizer.pad_token = untrained_tokenizer.eos_token

# Define the text generation function for GPT-2
def generate_with_gpt2(commentary_text, max_length, temperature):
    # Generate text using the finetuned model
    inputs = trained_tokenizer(commentary_text, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    trained_output = trained_model.generate(
        input_ids, 
        max_length=max_length, 
        num_beams=5, 
        do_sample=True, 
        temperature=temperature, 
        attention_mask=attention_mask,
        pad_token_id=trained_tokenizer.eos_token_id
    )
    trained_text = trained_tokenizer.decode(trained_output[0], skip_special_tokens=True)
    
    # Generate text using the base model
    inputs = untrained_tokenizer(commentary_text, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    untrained_output = untrained_model.generate(
        input_ids, 
        max_length=max_length, 
        num_beams=5, 
        do_sample=True, 
        temperature=temperature, 
        attention_mask=attention_mask,
        pad_token_id=untrained_tokenizer.eos_token_id
    )
    untrained_text = untrained_tokenizer.decode(untrained_output[0], skip_special_tokens=True)
    
    return trained_text, untrained_text

# Define the combined function for Gradio interface
def generate_with_all_models(commentary_text, num_words, max_length, temperature):
    lstm_output = generate_with_lstm(commentary_text, num_words)
    rnn_output = generate_with_rnn(commentary_text, num_words)
    gpt2_finetuned_output, gpt2_base_output = generate_with_gpt2(commentary_text, max_length, temperature)
    return lstm_output, rnn_output, gpt2_finetuned_output, gpt2_base_output

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_with_all_models,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter commentary text here...", label="Prompt"),
        gr.Slider(minimum=1, maximum=50, step=1, value=10, label="Number of words to predict (LSTM/RNN)"),
        gr.Slider(minimum=10, maximum=100, value=50, step=1, label="Max Length (GPT-2)"),        
        gr.Slider(minimum=0.01, maximum=1.99, value=0.7, label="Temperature (GPT-2)")
    ],
    outputs=[
        gr.Textbox(label="LSTM Model Output"),
        gr.Textbox(label="RNN Model Output"),
        gr.Textbox(label="GPT-2 Finetuned Model Output"), 
        gr.Textbox(label="GPT-2 Base Model Output")
    ],
    title="Text Generation with LSTM, RNN, and GPT-2 Models",
    description="Start writing a cricket commentary and the models will continue it. Compare outputs from LSTM, RNN, and GPT-2 (finetuned and base) models."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
