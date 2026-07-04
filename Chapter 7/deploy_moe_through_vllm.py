from vllm import LLM, SamplingParams  

# Define sampling parameters for generation  
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)  

# Initialize the MoE model 
llm = LLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", trust_remote_code=True, tensor_parallel_size=4) # Adjust to match your available GPU count  

# Generate completions for multiple concurrent prompts  
prompts = [  
    "Explain the importance of using MoE models on my local machine.",  
    "Compare sparse MoE layers to dense transformer networks."  
]  
outputs = llm.generate(prompts, sampling_params)  

# Print the outputs  
for output in outputs:  
    prompt = output.prompt  
    generated_text = output.outputs[0].text  
    print(f"Prompt: {prompt!r}\nGenerated: {generated_text!r}\n") 