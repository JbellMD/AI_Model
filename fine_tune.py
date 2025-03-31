from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

# Load the model and tokenizer
model_path = "C:/AI_Model/tinyllama_1.1B"
print("🚀 Loading model for fine-tuning...")
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Apply LoRA configuration
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# Load dataset
dataset = load_from_disk("C:/AI_Model/dataset/")

# Fine-tuning configuration
training_args = TrainingArguments(
    output_dir="C:/AI_Model/results",
    num_train_epochs=6,
    per_device_train_batch_size=1,
    save_steps=10,
    save_total_limit=1,
    logging_dir="C:/AI_Model/logs",
    remove_unused_columns=False  # ✅ Ensures prompt/response columns are kept
)


# Fine-tuning process
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
print("🚀 Starting fine-tuning...")
trainer.train()

# Save fine-tuned model
fine_tuned_path = "C:/AI_Model/fine_tuned_model"
model.save_pretrained(fine_tuned_path)
tokenizer.save_pretrained(fine_tuned_path)
print(f"✅ Fine-tuning complete! Model saved at {fine_tuned_path}")
