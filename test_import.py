"""
Test basic OpenAI import
"""

import sys
print(f"Python version: {sys.version}")

try:
    import openai
    print(f"OpenAI version: {openai.__version__}")
    print("Import successful")
    
    # Try creating client
    from openai import OpenAI
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    # Create client with minimal arguments
    client = OpenAI(api_key=api_key)
    print("Client created successfully")
    
except Exception as e:
    print(f"ERROR during import/init: {e}")
    import traceback
    traceback.print_exc()