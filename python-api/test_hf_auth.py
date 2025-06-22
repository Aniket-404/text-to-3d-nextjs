from huggingface_hub import login, whoami
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get Hugging Face token
huggingface_token = os.environ.get("HUGGINGFACE_API_KEY")
print(f"Hugging Face token: {huggingface_token[:5]}...{huggingface_token[-5:]}")

try:
    # Login with token
    login(token=huggingface_token)
    print("Login successful!")
    
    # Verify token and get user info
    user_info = whoami()
    print(f"Successfully authenticated as: {user_info['name']}")
    print(f"Email: {user_info.get('email', 'Not available')}")
    
except Exception as e:
    print(f"Authentication failed: {str(e)}")
