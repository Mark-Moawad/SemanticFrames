import os
from PyPDF2 import PdfReader

RESOURCE_DIR = "resources"
TXT_DIR = "resources_txt"
os.makedirs(TXT_DIR, exist_ok=True)

for filename in os.listdir(RESOURCE_DIR):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(RESOURCE_DIR, filename)
        txt_path = os.path.join(TXT_DIR, filename.replace(".pdf", ".txt"))
        print(f"Extracting text from {filename}...")
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved to {txt_path}")
print("All resource PDFs processed.")
