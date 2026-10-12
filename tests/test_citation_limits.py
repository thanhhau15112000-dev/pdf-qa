import pytest
from pydantic import ValidationError

from app.models import AnswerItem, Citation


def citation_set(count):
    return [Citation(page=(index % 30) + 1, quote=f"A verified source excerpt for citation {index + 1}.") for index in range(count)]


def test_answer_item_accepts_one_citation_per_supported_pdf_page():
    item = AnswerItem(
        title="Summary",
        status="supported",
        answer="Summary of the document.",
        citations=citation_set(10),
    )

    assert len(item.citations) == 10


def test_answer_item_keeps_citation_output_bounded():
    with pytest.raises(ValidationError):
        AnswerItem(
            title="Summary",
            status="supported",
            answer="Summary of the document.",
            citations=citation_set(31),
        )
