import os
import fitz  # PyMuPDF

PDF_DIR = "norms"
TXT_DIR = "norms_txt"
os.makedirs(TXT_DIR, exist_ok=True)

for filename in os.listdir(PDF_DIR):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, filename)
        txt_path = os.path.join(TXT_DIR, filename.replace(".pdf", ".txt"))
        print(f"Extracting text from {filename}...")
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved to {txt_path}")
print("All PDFs processed.")
