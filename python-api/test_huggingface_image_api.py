"""
Test the Hugging Face API image generation.
This script tests the Hugging Face API for image generation.
"""

import os
from dotenv import load_dotenv
from PIL import Image
import io
from huggingface_hub import InferenceClient
import time

def test_huggingface_image_generation():
    # Load environment variables
    load_dotenv()
    
    # Get Hugging Face token
    huggingface_token = os.environ.get("HUGGINGFACE_API_KEY")
    if not huggingface_token:
        print("Error: HUGGINGFACE_API_KEY not found in environment variables.")
        return False
    
    # Get model ID
    model_id = os.environ.get("SD_MODEL_ID", "stabilityai/stable-diffusion-3-medium-diffusers")
    
    try:
        # Print token and model info
        print(f"Using token: {huggingface_token[:5]}...{huggingface_token[-5:]}")
        print(f"Using model: {model_id}")
        
        # Initialize the Hugging Face Inference client
        print("Initializing InferenceClient...")
        client = InferenceClient(token=huggingface_token)
        
        # Test prompt
        prompt = "A beautiful mountain landscape with a lake, digital art"
        print(f"Generating image for prompt: \"{prompt}\"")
        
        # Start timer
        start_time = time.time()
        
        # Generate the image
        image = client.text_to_image(
            prompt=prompt,
            model=model_id,
            negative_prompt="low quality, bad anatomy, worst quality, low resolution, blurry",
            num_inference_steps=25,
            guidance_scale=7.0,
            width=512,  # Smaller for faster testing
            height=512
        )
        
        # End timer
        end_time = time.time()
        duration = end_time - start_time
        
        if image:
            print(f"Successfully generated image in {duration:.2f} seconds!")
            
            # Save the image
            output_path = "test_huggingface_output.png"
            image.save(output_path)
            print(f"Saved image to {output_path}")
            
            # Display image info
            print(f"Image size: {image.size}")
            print(f"Image format: {image.format}")
            
            return True
        else:
            print("Error: Image generation returned None")
            return False
            
    except Exception as e:
        print(f"Error: Failed to generate image with Hugging Face API: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_huggingface_image_generation()
    print(f"Test {'succeeded' if success else 'failed'}")
