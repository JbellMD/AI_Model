from datasets import load_dataset
from transformers import AutoTokenizer

# Load JSON dataset
dataset = load_dataset("json", data_files="C:/AI_Model/dataset/apophatic_psychology.json")["train"]


# Load tokenizer
model_path = "C:/AI_Model/tinyllama_1.1B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Preprocessing function
def preprocess_data(example):
    input_text = "### Question:\n" + example["prompt"] + "\n\n### Answer:\n" + example["response"]
    tokenized = tokenizer(input_text, padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Labels are the same as input (causal LM)
    return tokenized

# Apply preprocessing
dataset = dataset.map(preprocess_data, remove_columns=["prompt", "response"])

# ✅ Save dataset correctly (NO `.from_dict()` needed)
dataset.save_to_disk("C:/AI_Model/dataset/")
print("✅ Dataset formatted and saved successfully!")



