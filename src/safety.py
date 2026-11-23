import re
UNSAFE_PATTERNS = [
    "the fetus definitely has",
    "this confirms spina bifida",
    "you definitely have",
    "no need to consult a doctor",
]

def check_response(text: str) -> dict:
    normalized = (text or "").lower()
    flagged = [p for p in UNSAFE_PATTERNS if p in normalized]
    return {
        "approved": not bool(flagged),
        "reason": "Potentially unsafe diagnostic wording detected." if flagged else "Basic phrase check passed.",
        "matches": flagged,
    }


def validate_question(question: str) -> tuple[bool, str]:
    if not question or not question.strip():
        return False, "Question cannot be empty."

    if len(question) > 2000:
        return False, "Question is too long (maximum 2,000 characters)."

    normalized = question.lower()

    unsafe_requests = [
        "diagnose my fetus",
        "does my baby have spina bifida",
        "confirm spina bifida from this image",
        "tell me what disease my fetus has",
        "prescribe medication",
        "what medicine should i take",
        "recommend a treatment for my pregnancy",
        "should i stop taking my medication",
    ]

    for pattern in unsafe_requests:
        if pattern in normalized:
            return (
                False,
                "PrenatalAI cannot provide patient-specific diagnoses "
                "or treatment recommendations. Please consult a "
                "qualified healthcare professional."
            )

    return True, "ok"


def validate_citations(answer: str, context: str) -> dict:
    answer_labels = set(re.findall(r"\[S(\d+)\]", answer or ""))
    context_labels = set(re.findall(r"\[S(\d+)\]", context or ""))

    invalid = sorted(answer_labels - context_labels)

    # Reject numbered citations such as [2] instead of [S2].
    unsupported_format = re.findall(
        r"(?<![A-Za-z])\[(\d+)\]",
        answer or ""
    )

    approved = not invalid and not unsupported_format

    return {
        "approved": approved,
        "reason": (
            "Citation references are valid."
            if approved
            else "Invalid citation reference or format detected."
        ),
        "invalid_citations": [f"[S{x}]" for x in invalid],
        "unsupported_format": [
            f"[{x}]" for x in unsupported_format
        ],
    }
