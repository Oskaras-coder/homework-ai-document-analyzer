import fitz # PyMuPDF - layout-aware PDF parser
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

#  https://pymupdf.readthedocs.io/en/latest/
def load_and_split(file):
    raw_docs = _extract_text_by_layout(file)  # Step 1: extract layout-aware text
    return _split_documents(raw_docs)         # Step 2: split into chunks


def _extract_text_by_layout(file):
    docs = []
    pdf = fitz.open(stream=file.read(), filetype="pdf")  # Open PDF from stream

    # Quick solution to extract text by layout
    for i, page in enumerate(pdf):  # Iterate through each page
        blocks = page.get_text("blocks")  # Extract text blocks with coordinates
        sorted_blocks = sorted(blocks, key=lambda b: (round(b[1] / 20), b[0])) # Sort blocks top-to-bottom, left-to-right for better logical reading order

        text = "\n".join(
            block[4].strip() for block in sorted_blocks if block[4].strip()
        )
        # Join non-empty block texts

        if text:
            docs.append(Document(page_content=text, metadata={"page": i})) # Store page as a LangChain Document with metadata

    return docs



def _split_documents(docs):  # Chunk the text for LLM Use
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)
