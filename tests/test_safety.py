from src.safety import check_response, validate_question

def test_empty_question_rejected():
    assert validate_question("")[0] is False

def test_normal_question_accepted():
    assert validate_question("What is a fetal ultrasound?")[0] is True

def test_diagnostic_claim_flagged():
    assert check_response("This confirms Spina Bifida.")["approved"] is False

def test_neutral_text_passes():
    assert check_response("The literature describes screening limitations.")["approved"] is True
