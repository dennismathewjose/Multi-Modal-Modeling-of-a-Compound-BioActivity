"""
Test OpenAI API connection
Verifies that the API key is valid and can make requests
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("ERROR: No API key found in .env file")
    exit(1)

print("\nTesting OpenAI API connection...")
print("-" * 70)
print(f"API key loaded (prefix: {api_key[:20]}...)")

# Test API call with correct initialization
try:
    from openai import OpenAI
    
    # Initialize client without extra parameters
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Reply with: API working"}],
        max_tokens=10
    )
    
    result = response.choices[0].message.content
    print(f"API Response: {result}")
    print("-" * 70)
    print("SUCCESS: Setup complete\n")
    
except Exception as e:
    print(f"ERROR: {e}")
    print("\nCheck:")
    print("1. API key is correct")
    print("2. Billing is set up at platform.openai.com")
    exit(1)