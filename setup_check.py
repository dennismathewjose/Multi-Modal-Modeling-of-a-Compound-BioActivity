import sys
import os

print("=" * 70)
print("ENVIRONMENT CHECK")
print("=" * 70)

# Check if in virtual environment
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print(f"\n Virtual environment active")
else:
    print(f"\n  Virtual environment NOT active")

# Check packages
print(f"\nPackages:")
packages = ['openai', 'fitz', 'PIL', 'dotenv', 'pandas', 'numpy']
for pkg in packages:
    try:
        __import__(pkg)
        print(f"   {pkg}")
    except ImportError:
        print(f"   {pkg}")

# Check folders
print(f"\nFolders:")
folders = ['vlm_extraction/papers', 'vlm_extraction/outputs', 'vlm_extraction/smiles_data', 'vlm_extraction/src']
for folder in folders:
    if os.path.exists(folder):
        print(f"   {folder}")
    else:
        print(f"   {folder}")

# Check .env
print(f"\nConfiguration:")
if os.path.exists('.env'):
    print(f"   .env file exists")
    from dotenv import load_dotenv
    load_dotenv()
    if os.getenv('OPENAI_API_KEY'):
        print(f"   API key loaded")
    else:
        print(f"   API key not found")
else:
    print(f"   .env file missing")

print("=" * 70)