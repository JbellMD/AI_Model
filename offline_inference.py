import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model & tokenizer
model_path = "C:/AI_Model/fine_tuned_model"
print("🚀 Loading model for offline inference...")
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_response(prompt, max_length=300):
    """Generates AI response from fine-tuned model."""
    input_text = f"### Question:\n{prompt}\n\n### Answer:\n"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(
        input_ids, 
        max_length=max_length, 
        num_return_sequences=1, 
        temperature=0.9, 
        top_k=50, 
        top_p=0.95, 
        do_sample=True, 
        repetition_penalty=1.1
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("📝 Offline AI Assistant is ready. Type 'exit' to quit.")
    while True:
        user_input = input("\n🧠 Enter your question: ")
        if user_input.lower() == "exit":
            break
        response = generate_response(user_input)
        print("\n🤖 AI Response:\n", response)

