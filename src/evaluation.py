from src.safety import check_response, validate_question

def run_smoke_tests():
    tests = []
    tests.append({"test": "non-empty question accepted", "passed": validate_question("What is prenatal ultrasound?")[0]})
    tests.append({"test": "empty question rejected", "passed": not validate_question("")[0]})
    tests.append({"test": "neutral educational text passes", "passed": check_response("The literature describes screening limitations.")["approved"]})
    tests.append({"test": "obvious diagnostic phrase flagged", "passed": not check_response("This confirms Spina Bifida.")["approved"]})
    return tests
