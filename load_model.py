import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# STEP 3: Load the LLaMA Model with LoRA
# Dynamically set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load LLaMA model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=False,  # Avoid bitsandbytes issues
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)  # Move model to device explicitly

# Configure LoRA for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Focus on specific attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

print("Model loaded and configured with LoRA.")
