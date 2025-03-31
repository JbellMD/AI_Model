from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model & tokenizer
model_path = "C:/AI_Model/fine_tuned_model"
print("🚀 Loading model for inference...")
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define inference function
def generate_response(prompt, max_length=300):
    input_text = f"### Question:\n{prompt}\n\n### Answer:\n"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(
        input_ids, 
        max_length=max_length, 
        num_return_sequences=1, 
        temperature=0.9,  # Increased for varied responses
        top_k=50, 
        top_p=0.95, 
        do_sample=True,  # Ensures sampling-based generation
        repetition_penalty=1.1  # Avoids repetitive outputs
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Test prompt
prompt = "What is the nature of the self?"
response = generate_response(prompt)

print("\n🧠 Model Response:\n", response)

