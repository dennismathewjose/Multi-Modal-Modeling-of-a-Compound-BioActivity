"""
PDF Processor - Extract pages and regions from PDF files
Handles conversion of PDF pages to high-resolution images
"""

import fitz
from PIL import Image
import io

class PDFProcessor:
    """
    Processes PDF files and extracts pages as images
    """
    
    def __init__(self, pdf_path):
        """
        Initialize PDF processor
        
        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = pdf_path
        self.doc = None
        self._load_pdf()
    
    def _load_pdf(self):
        """Load PDF document"""
        try:
            self.doc = fitz.open(self.pdf_path)
            print(f"Loaded PDF: {self.pdf_path}")
            print(f"Total pages: {len(self.doc)}")
        except Exception as e:
            raise Exception(f"Failed to load PDF: {e}")
    
    def extract_page(self, page_num, dpi=300):
        """
        Extract entire page as image
        
        Args:
            page_num: Page number (0-indexed)
            dpi: Resolution in dots per inch
            
        Returns:
            PIL Image object
        """
        if page_num < 0 or page_num >= len(self.doc):
            raise ValueError(f"Invalid page number: {page_num}")
        
        page = self.doc.load_page(page_num)
        
        # Convert DPI to zoom factor
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        print(f"Extracted page {page_num + 1} at {dpi} DPI")
        print(f"Image size: {img.size}")
        
        return img
    
    def save_image(self, image, output_path):
        """
        Save image to file
        
        Args:
            image: PIL Image object
            output_path: Path to save image
        """
        image.save(output_path, "PNG")
        print(f"Saved image to: {output_path}")
    
    def close(self):
        """Close PDF document"""
        if self.doc:
            self.doc.close()
            print("PDF closed")

# Test the processor
if __name__ == "__main__":
    # Test extraction of Table 1 (page 41, index 40)
    pdf_path = "/Users/dennis_m_jose/Documents/GitHub/Multi-Modal-Modeling-of-a-Compound-BioActivity/vlm_extraction/papers/egfr_paper.pdf"
    processor = PDFProcessor(pdf_path)
    
    # Extract page 41 (0-indexed as 40)
    table_image = processor.extract_page(40, dpi=300)
    
    # Save to outputs folder
    processor.save_image(table_image, "../outputs/table1.png")
    
    processor.close()
    
    print("\nTable 1 extracted successfully")