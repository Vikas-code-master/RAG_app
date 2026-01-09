import PyPDF2
import os
from pathlib import Path

def extract_pdf_to_txt(pdf_path: str, output_dir: str = "extracted_pages") -> None:
    """
    Extract text from a PDF file and save each page as a separate .txt file.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted text files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Open and read the PDF
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        
        # Extract text from each page
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            # Save to individual txt file
            output_file = os.path.join(output_dir, f"page_{page_num + 1}.txt")
            with open(output_file, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)
            
            print(f"Extracted page {page_num + 1} to {output_file}")


if __name__ == "__main__":
    # Example usage
    pdf_file = "Think-And-Grow-Rich_2011-06.pdf"  # Replace with your PDF path
    extract_pdf_to_txt(pdf_file)