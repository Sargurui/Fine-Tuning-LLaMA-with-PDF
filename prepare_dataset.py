from datasets import Dataset
from transformers import AutoTokenizer

# Load tokenizer for LLaMA
model_name = "meta-llama/Llama-3.2-3B"  # Replace with the actual model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the pad_token to eos_token or add a new pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

print("Tokenizer loaded and padding token set.")

# Combine extracted text into a Hugging Face dataset
data = {"text": pdf_texts}
dataset = Dataset.from_dict(data)

print("Dataset created.")

# Tokenize the dataset
def tokenize_function(example):
    encodings = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    # Ensure labels are properly aligned for language modeling
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

tokenized_dataset = dataset.map(tokenize_function, batched=True)
