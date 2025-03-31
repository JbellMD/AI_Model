from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Use the correct TinyLlama model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load the model and tokenizer
print("🚀 Loading TinyLlama-1.1B-Chat-v1.0...")
model = AutoModelForCausalLM.from_pretrained(model_name)
print("✅ Model loaded successfully!")

print("🚀 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✅ Tokenizer loaded successfully!")

# Save the model and tokenizer
save_path = "C:/AI_Model/tinyllama_1.1B"
print(f"💾 Saving model to {save_path} ...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("✅ Model and tokenizer saved successfully!")

print("🎯 Execution complete. Ready for dataset preparation!")


