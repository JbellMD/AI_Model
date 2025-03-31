import tkinter as tk
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model & tokenizer
model_path = "C:/AI_Model/fine_tuned_model"
print("🚀 Loading model for GUI-based offline inference...")
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_response():
    """Generates AI response from fine-tuned model."""
    prompt = input_box.get("1.0", "end-1c")
    if not prompt.strip():
        return
    input_text = f"### Question:\n{prompt}\n\n### Answer:\n"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(
        input_ids, 
        max_length=300, 
        num_return_sequences=1, 
        temperature=0.9, 
        top_k=50, 
        top_p=0.95, 
        do_sample=True, 
        repetition_penalty=1.1
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    output_box.config(state="normal")
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, response)
    output_box.config(state="disabled")

# GUI Setup
root = tk.Tk()
root.title("Offline AI Assistant")

tk.Label(root, text="Enter Your Question:").pack()
input_box = tk.Text(root, height=5, width=50)
input_box.pack()

generate_button = tk.Button(root, text="Generate Response", command=generate_response)
generate_button.pack()

tk.Label(root, text="AI Response:").pack()
output_box = tk.Text(root, height=10, width=50, state="disabled")
output_box.pack()

root.mainloop()
