"""
Validate the Hugging Face API token.
This script checks if the provided Hugging Face API token is valid.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import login, whoami
import sys

def validate_huggingface_token():
    # Load environment variables
    load_dotenv()
    
    # Get Hugging Face token
    huggingface_token = os.environ.get("HUGGINGFACE_API_KEY")
    if not huggingface_token:
        print("Error: HUGGINGFACE_API_KEY not found in environment variables.")
        return False
    
    try:
        # Login with the token
        print(f"Attempting to login with token: {huggingface_token[:5]}...{huggingface_token[-5:]}")
        login(token=huggingface_token)
        
        # Verify token and get user info
        user_info = whoami()
        print(f"Success! Logged in to Hugging Face Hub as {user_info['name']}")
        return True
    except Exception as e:
        print(f"Error: Failed to login with Hugging Face token: {str(e)}")
        return False

if __name__ == "__main__":
    success = validate_huggingface_token()
    sys.exit(0 if success else 1)
