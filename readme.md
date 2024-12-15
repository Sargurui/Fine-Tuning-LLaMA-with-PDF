# Fine-Tuning LLaMA with PDF Text Data

This project fine-tunes the LLaMA model on text extracted from PDF files using LoRA (Low-Rank Adaptation) for efficient fine-tuning.

## Project Overview

The workflow is as follows:
1. **Extract Text**: Extract text from PDF files using pdfplumber.
2. **Prepare Dataset**: Convert extracted text into a Hugging Face dataset and tokenize it.
3. **Load LLaMA Model with LoRA**: Load the LLaMA language model and configure LoRA for parameter-efficient fine-tuning.
4. **Fine-Tune the Model**: Fine-tune the model with the prepared dataset using Hugging Face's Trainer API.
5. **Test the Fine-Tuned Model**: Generate text from the fine-tuned model.

## Project Structure

```
├── extract_text.py       # Extract text from PDF files
├── prepare_dataset.py    # Prepare dataset and tokenize it
├── load_model.py         # Load LLaMA model with LoRA configuration
├── fine_tune.py          # Fine-tune the model
├── test_model.py         # Test the fine-tuned model
├── requirements.txt      # Required libraries
└── README.md             # Project documentation
```

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/llama-fine-tuning
cd llama-fine-tuning
```

### 2. Install Dependencies

Install the required libraries listed in requirements.txt:

```bash
pip install -r requirements.txt
```

Libraries:
- torch
- transformers
- peft
- accelerate
- datasets
- pdfplumber

### 3. Prepare the PDF Files

Place your PDF files in a folder, e.g., `pdf_file/`. Update the file paths in `extract_text.py`:

```python
pdf_files = [
    "pdf_file/document1.pdf", 
    "pdf_file/document2.pdf"
]
```

## Running the Project

Follow these steps in order:

### Extract Text

Run `extract_text.py` to extract text from PDF files:

```bash
python extract_text.py
```

### Prepare the Dataset

Run `prepare_dataset.py` to tokenize and create a Hugging Face dataset:

```bash
python prepare_dataset.py
```

### Load the Model

Run `load_model.py` to load the LLaMA model and configure LoRA:

```bash
python load_model.py
```

### Fine-Tune the Model

Run `fine_tune.py` to fine-tune the model:

```bash
python fine_tune.py
```

### Test the Model

Run `test_model.py` to test the fine-tuned model:

```bash
python test_model.py
```

## Output

- **Fine-tuned model**: Saved in the `./fine_tuned_llama` directory.
- **Logs and checkpoints**: Saved in the `./results` directory.
- **Sample Output**: Text generated using the fine-tuned model.

## Requirements

- Python 3.8+
- GPU (recommended) for faster training
- Libraries from `requirements.txt`

## Credits

- Hugging Face Transformers: https://huggingface.co/
- PEFT Library: https://github.com/huggingface/peft
- pdfplumber: https://github.com/jsvine/pdfplumber