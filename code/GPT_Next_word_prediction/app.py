from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr
import torch

trained_tokenizer = GPT2Tokenizer.from_pretrained("Kumarkishalaya/GPT-2-next-word-prediction")
trained_model = GPT2LMHeadModel.from_pretrained("Kumarkishalaya/GPT-2-next-word-prediction")
untrained_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
untrained_model = GPT2LMHeadModel.from_pretrained("gpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"
trained_model.to(device)
untrained_model.to(device)

def generate(commentary_text, max_length, temperature):
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

# Create Gradio interface
iface = gr.Interface(
    fn=generate, 
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt"),
        gr.Slider(minimum=10, maximum=100, value=50, step=1,label="Max Length"),        
        gr.Slider(minimum=0.01, maximum=1.99, value=0.7, label="Temperature")
    ], 
    outputs=[
        gr.Textbox(label="commentary generation from finetuned GPT2 Model"), 
        gr.Textbox(label="commentary generation from base GPT2 Model")
    ],
    title="GPT-2 Text Generation",
    description="start writing a cricket commentary and GPT-2 will continue it using both a finetuned and base model."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()