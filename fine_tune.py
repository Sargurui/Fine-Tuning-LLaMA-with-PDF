from transformers import TrainingArguments, Trainer

# STEP 4: Fine-Tune the Model
# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Reduce batch size if memory issues occur
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,  # Keep only the last 2 checkpoints
    fp16=torch.cuda.is_available(),  # Enable mixed precision if CUDA is available
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("Starting fine-tuning...")

# Fine-tune the model
trainer.train()

print("Fine-tuning completed.")

# STEP 5: Save the Fine-Tuned Model
model.save_pretrained("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")

print("Fine-tuned model saved.")
