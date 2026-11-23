from rag.retrieval import answer_question

def test_retrieval_returns_sources():
    result = answer_question("What is the banana sign?")
    assert isinstance(result["answer"], str)
    assert len(result["sources"]) > 0
