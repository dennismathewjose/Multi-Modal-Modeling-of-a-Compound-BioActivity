"""
VLM Extractor - Extract structured data from images using GPT-4V
Processes table images and returns structured JSON data
"""

import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VLMExtractor:
    """
    Extracts structured data from images using Vision-Language Models
    """
    
    def __init__(self, api_key=None, model="gpt-4o"):
        """
        Initialize VLM extractor
        
        Args:
            api_key: OpenAI API key (defaults to env variable)
            model: Model to use for extraction
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        
        print(f"VLM Extractor initialized with model: {model}")
    
    def _encode_image(self, image_path):
        """
        Encode image to base64
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_table_data(self, image_path, prompt=None):
        """
        Extract structured data from table image
        
        Args:
            image_path: Path to table image
            prompt: Custom extraction prompt (uses default if None)
            
        Returns:
            Dictionary with extracted data
        """
        # Default prompt for EGFR table extraction
        if prompt is None:
            prompt = self._get_default_prompt()
        
        # Encode image
        print(f"Encoding image: {image_path}")
        base64_image = self._encode_image(image_path)
        
        # Call API
        print("Sending request to GPT-4V...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000
            )
            
            # Extract response
            result_text = response.choices[0].message.content
            print("Received response from GPT-4V")
            
            # Parse JSON
            result = self._parse_json(result_text)
            
            return result
            
        except Exception as e:
            raise Exception(f"VLM extraction failed: {e}")
    
    def _get_default_prompt(self):
        """
        Get default extraction prompt for EGFR table
        
        Returns:
            Prompt string
        """
        return """
Extract all compound IDs and their IC50 values from this table.

Table structure:
- Row headers: Compound IDs (2f, 18, 3a, 3b, 3c, 3d, 3e, 3f, 8a, 8b, 8c, 8d, 8e)
- Column headers: EGFR variants (WT, Del19, d747-752/P753S, L858R, L858R/T790M)
- Values: IC50 in nM

Special handling:
1. "n.d." → return null
2. "> 100000" or ">10000" → return the numeric value (100000 or 10000)
3. "1795.3 ± 859.1" → return mean only (1795.3)
4. Empty cells → return null

Output format (JSON):
{
  "2f": {
    "ic50_wt": 1795.3,
    "ic50_del19": 18.8,
    "ic50_d747_752_p753s": 0.4,
    "ic50_l858r": null,
    "ic50_l858r_t790m": 1338.0
  },
  "3a": { ... }
}

Return ONLY the JSON object, no markdown formatting or explanation.
"""
    
    def _parse_json(self, text):
        """
        Parse JSON from response text
        
        Args:
            text: Response text that may contain JSON
            
        Returns:
            Parsed JSON dictionary
        """
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Parse JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Response text: {text[:200]}...")
            raise

# Test the extractor
if __name__ == "__main__":
    # Initialize extractor
    extractor = VLMExtractor()
    
    # Extract data from Table 1
    image_path = "../outputs/table1.png"
    extracted_data = extractor.extract_table_data(image_path)
    
    # Save to JSON file
    output_path = "../outputs/extracted_data.json"
    with open(output_path, 'w') as f:
        json.dump(extracted_data, f, indent=2)
    
    print(f"\nExtracted data saved to: {output_path}")
    print(f"Total compounds extracted: {len(extracted_data)}")