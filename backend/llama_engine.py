from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", 
    use_auth_token="llama2-access"  # Updated to `use_auth_token`
)

# Set up the quantization configuration
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

# Load the model with device_map to handle large model loading across devices
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",  # This automatically uses the available devices (GPU or CPU)
    use_auth_token="llama2-access",  # Correctly using the token parameter
    quantization_config=quant_config,
    max_memory={0: "90GB"}  # Adjust memory usage for large models
)

def generate_response(user_text, emotion):
    system_prompt = f"You are an empathetic mental health support assistant. The user is feeling {emotion}."
    full_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_text} [/INST]"

    # Tokenize the input text and move to the same device as the model
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Generate the response with the specified parameters
        output = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)

    # Decode and return the response, stripping unnecessary tokens
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()

