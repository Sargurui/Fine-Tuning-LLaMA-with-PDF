import pdfplumber

# STEP 1: Extract Text from PDFs
def extract_text_from_pdfs(pdf_files):
    text_data = []
    for pdf_file in pdf_files:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            text_data.append(text)
    return text_data

# List of PDF files (update with your file paths)
pdf_files = [
    "pdf_file/document1.pdf", 
    "pdf_file/document2.pdf", 
    "pdf_file/document3.pdf", 
    "pdf_file/document4.pdf", 
    "pdf_file/document5.pdf", 
    "pdf_file/document6.pdf"
]

# Extract text from PDFs
pdf_texts = extract_text_from_pdfs(pdf_files)
