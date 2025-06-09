import io
import json
from unittest.mock import patch
from main import process_document
from core.llm import get_llm
from langchain.schema import Document

DUMMY_PDF = b"%PDF-1.4\n%Dummy PDF content for testing"

DUMMY_SUMMARY = {"output_text": "This is a summary of the document."}
DUMMY_QA = {
    "What is the main topic?": {"result": "AI and document analysis."},
    "Who is the intended audience?": {
        "result": "Software developers and data analysts."
    },
}


class DummyFile:
    def __init__(self, content):
        self.content = content

    def read(self):
        return self.content


@patch("main._run_pipeline")
def test_process_document_creates_json_and_returns_data(mock_pipeline, tmp_path):
    dummy_file = DummyFile(DUMMY_PDF)
    dummy_docs = [Document(page_content="This is page text", metadata={"page": 0})]
    mock_pipeline.return_value = {"summary": DUMMY_SUMMARY, "qa": DUMMY_QA}

    with patch("main._prepare_documents", return_value=dummy_docs), patch(
        "main._prepare_retriever"
    ):
        json_path = tmp_path / "output.json"
        summary, qa = process_document(dummy_file, str(json_path))

    assert summary == DUMMY_SUMMARY
    assert qa == DUMMY_QA
    assert json_path.exists()

    with open(json_path, encoding="utf-8") as f:
        saved = json.load(f)
        assert saved["summary"] == DUMMY_SUMMARY
        assert saved["qa"] == DUMMY_QA


def test_model_switching_logic_real():
    llm = get_llm()
    class_name = llm.__class__.__name__.lower()

    if "openai" in class_name:
        assert "openai" in class_name
    else:
        assert "ollama" in class_name
