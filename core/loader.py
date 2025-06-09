import pdfplumber
import fitz
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split(file):
    docs = []

    # Read PDF file using PyMuPDF
    pdf = fitz.open(stream=file.read(), filetype="pdf")

    for i, page in enumerate(pdf):
        blocks = page.get_text("blocks")  # List of text blocks: (x0, y0, x1, y1, "text", block_no, block_type)
        
        # Sort blocks: first by vertical position, then horizontal (i.e., top-to-bottom, left-to-right)
        sorted_blocks = sorted(blocks, key=lambda b: (round(b[1] / 20), b[0]))

        text = "\n".join(block[4].strip() for block in sorted_blocks if block[4].strip())
        if text:
            docs.append(Document(page_content=text, metadata={"page": i}))

    # Split after layout-aware extraction
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)