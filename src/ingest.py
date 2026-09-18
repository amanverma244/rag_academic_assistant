import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF

class PDFIngestor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.filename = os.path.basename(pdf_path)

    def extract_text_with_metadata(self):
        """Extracts text page-by-page from a PDF file."""
        doc = fitz.open(self.pdf_path)
        pages_data = []
        
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text")
            if text.strip(): 
                pages_data.append({
                    "page": page_num + 1,
                    "text": text,
                    "source": self.filename
                })
        return pages_data

    def chunk_documents(self, pages_data, chunk_size=1000, chunk_overlap=200):
        """Splits pages into overlapping chunks while preserving metadata."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        docs = []
        for page_info in pages_data:
            splits = text_splitter.split_text(page_info["text"])
            for split in splits:
                docs.append({
                    "content": split,
                    "metadata": {
                        "source": page_info["source"],
                        "page": page_info["page"]
                    }
                })
        return docs

if __name__ == "__main__":
    
    sample_pdf = "data/raw_pdfs/sample.pdf"
    if os.path.exists(sample_pdf):
        ingestor = PDFIngestor(sample_pdf)
        pages = ingestor.extract_text_with_metadata()
        chunks = ingestor.chunk_documents(pages)
        print(f"Successfully extracted and split {len(chunks)} chunks from {sample_pdf}.")
    else:
        print(f"Please place a sample PDF at {sample_pdf} to test ingestion.")


