"""
Verify PDF is readable and count pages
"""

import fitz

pdf_path = "/Users/dennis_m_jose/Documents/GitHub/Multi-Modal-Modeling-of-a-Compound-BioActivity/vlm_extraction/papers/egfr_paper.pdf"

try:
    doc = fitz.open(pdf_path)
    print(f"PDF loaded successfully")
    print(f"Total pages: {len(doc)}")
    print(f"Table 1 should be on page 41 (index 40)")
    doc.close()
except Exception as e:
    print(f"ERROR: {e}")