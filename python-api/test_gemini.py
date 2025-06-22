"""
Test script for image generation using Google's Gemini API
"""

import os
import sys
import logging
import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_gemini_generation():
    """Test generating an image with Gemini"""
    logger.info("Testing Gemini image generation...")
    
    try:
        # Check if the Gemini API key is set
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            return False
        
        logger.info("Configuring Gemini API...")
        genai.configure(api_key=gemini_key)
        
        # Initialize Gemini Pro Vision model
        model = genai.GenerativeModel('gemini-1.5-pro')
        logger.info("Successfully initialized Gemini model")
        
        # Test with a simple prompt
        prompt = "A beautiful mountain landscape with a lake, sunset, detailed, photorealistic"
        
        # Generate the image
        logger.info(f"Generating image with prompt: {prompt}")
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.9,
                "top_p": 1,
                "top_k": 32,
                "max_output_tokens": 2048,
            },
            stream=False
        )
        
        # Check if generation was successful
        if response and hasattr(response, 'candidates') and response.candidates:
            logger.info("Successfully generated content with Gemini!")
            
            # Get the image parts from the response
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'image') and part.image:
                    # Convert the image data to PIL Image
                    image_data = BytesIO(part.image.data)
                    image = Image.open(image_data)
                    
                    # Save the image
                    output_path = "gemini_test_output.png"
                    image.save(output_path)
                    logger.info(f"Image saved as {output_path}")
                    
                    # Display image dimensions
                    logger.info(f"Image size: {image.width}x{image.height}")
                    return True
            
            logger.error("No image found in the response")
            return False
        else:
            logger.error("Failed to generate image with Gemini.")
            return False
            
    except Exception as e:
        logger.error(f"Error during Gemini testing: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_gemini_generation()
    if success:
        logger.info("Gemini test completed successfully!")
        sys.exit(0)
    else:
        logger.error("Gemini test failed.")
        sys.exit(1)
