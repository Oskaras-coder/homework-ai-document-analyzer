import fitz
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_split(file):
    raw_docs = _extract_text_by_layout(file)
    return _split_documents(raw_docs)


def _extract_text_by_layout(file):
    docs = []
    pdf = fitz.open(stream=file.read(), filetype="pdf")

    for i, page in enumerate(pdf):
        blocks = page.get_text("blocks")
        sorted_blocks = sorted(blocks, key=lambda b: (round(b[1] / 20), b[0]))
        text = "\n".join(
            block[4].strip() for block in sorted_blocks if block[4].strip()
        )
        if text:
            docs.append(Document(page_content=text, metadata={"page": i}))

    return docs


def _split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)
