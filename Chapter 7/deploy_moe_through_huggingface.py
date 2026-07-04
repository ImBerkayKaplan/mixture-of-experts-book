from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator

# Load the tokenizer and model 
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1") 
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1") 

# Prepare the input 
prompt = "Explain the importance of using MoE models on my local machine." 
inputs = tokenizer(prompt, return_tensors="pt") 

# Generate the output 
outputs = model.generate(**inputs) 
print(tokenizer.decode(outputs[0], skip_special_tokens=True)) 

accelerator = Accelerator() 
model = AutoModelForCausalLM.from_pretrained( 
    "mistralai/Mixtral-8x7B-v0.1", 
    torch_dtype="auto" 
) 

# Config for 8-bit precision #A 
bnb_config_8bit = BitsAndBytesConfig( 
    load_in_8bit=True
) 

# Load the model in 8-bit precision
model_8bit = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    quantization_config=bnb_config_8bit,
    device_map="auto"
) 

# Config for 4-bit precision
bnb_config_4bit = BitsAndBytesConfig( 
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
) 

# Load the model in 4-bit precision
model_4bit = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    quantization_config=bnb_config_4bit,
    device_map="auto"
)