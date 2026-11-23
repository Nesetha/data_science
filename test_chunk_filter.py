
from rag.vector_store import (
    is_reference_chunk,
    is_metadata_chunk,
    is_low_quality_chunk
)

samples = {
    "Metadata chunk": """
    ultrasound Spina Bifida
    Accepted: 21 February 2025
    Published online: 6 March 2025
    Copyright The Author(s) 2025
    Novel Techniques in Antenatal Imaging
    """,

    "Research chunk": """
    Spinal dysraphism is a range of congenital spinal
    defects arising from incomplete neural tube closure
    during embryonic development. Prenatal ultrasound
    provides initial detection of these abnormalities.
    """,

    "Reference chunk": """
    1. Smith et al. Journal of Medicine, vol. 12, pp. 20-25.
    2. Jones et al. DOI: doi.org/example
    """
}

for name, text in samples.items():
    print("\n", name)
    print("Reference:", is_reference_chunk(text))
    print("Metadata:", is_metadata_chunk(text))
    print("Low quality:", is_low_quality_chunk(text))
