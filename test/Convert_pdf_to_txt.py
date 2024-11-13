

import os
from pdfminer.high_level import extract_text

# Specify the root directory containing the folders
root_dir = os.getcwd()  # Current working directory  # Replace with the path to your folders

def convert_pdf_to_txt(pdf_path, txt_path):
    try:
        # Extract text from PDF
        text = extract_text(pdf_path)
        # Write the extracted text to a .txt file
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
    except Exception as e:
        print(f"Failed to convert {pdf_path}: {e}")

# Iterate over folders and files
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.pdf'):
                pdf_path = os.path.join(folder_path, file_name)
                txt_path = os.path.splitext(pdf_path)[0] + '.txt'
                convert_pdf_to_txt(pdf_path, txt_path)
                print(f"Converted {pdf_path} to {txt_path}")

