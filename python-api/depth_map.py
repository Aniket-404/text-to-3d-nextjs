"""
Depth map generation for 2D images.
This module provides functionality to generate depth maps from 2D images
and convert them to 3D models (OBJ files).
"""

import os
import requests
import io
from io import BytesIO
import base64
import logging
import numpy as np
import time
import tempfile
from PIL import Image
import cloudinary
import cloudinary.uploader
from urllib.parse import urlparse
from dotenv import load_dotenv
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import open3d as o3d
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
)

# Initialize the DPT model and processor
try:
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-beit-large-512")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-large-512")
    logger.info("Successfully initialized DPT model")
except Exception as e:
    logger.error(f"Failed to initialize DPT model: {str(e)}")
    raise

def download_image(image_url):
    """Download an image from a URL"""
    logger.info(f"Downloading image from {image_url}")
    response = requests.get(image_url, timeout=30)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise ValueError(f"Failed to download image: {response.status_code}")

def generate_depth_map(image):
    """Generate a depth map from an image using DPT model"""
    logger.info("Generating depth map using DPT")
    
    try:
        # Prepare image for the model
        inputs = processor(images=image, return_tensors="pt")
        
        # Generate depth prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # Convert to numpy array and normalize
        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = (depth_map * 255 / np.max(depth_map)).astype("uint8")
        
        return Image.fromarray(depth_map)
    except Exception as e:
        logger.error(f"Failed to generate depth map: {str(e)}")
        raise

def create_point_cloud(image, depth_map, fx=1000, fy=1000):
    """Create a point cloud from an image and its depth map"""
    logger.info("Creating point cloud")
    
    # Convert inputs to numpy arrays
    depth_array = np.array(depth_map)
    color_array = np.array(image)
    
    # Calculate camera parameters
    cx = color_array.shape[1] / 2
    cy = color_array.shape[0] / 2
    
    # Generate point cloud
    point_cloud = []
    colors = []
    for v in range(depth_array.shape[0]):
        for u in range(depth_array.shape[1]):
            Z = depth_array[v, u]
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            point_cloud.append([X, Y, Z])
            colors.append(color_array[v, u])
    
    return np.array(point_cloud), np.array(colors) / 255.0

def create_obj_file(point_cloud, colors):
    """Create an OBJ file from point cloud data"""
    logger.info("Creating OBJ file")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create a mesh from the point cloud
    try:
        # Estimate normals
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        
        # Create mesh using Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp_file:
            o3d.io.write_triangle_mesh(tmp_file.name, mesh)
            return tmp_file.name
    except Exception as e:
        logger.error(f"Failed to create OBJ file: {str(e)}")
        raise

def upload_to_cloudinary(file_path, public_id):
    """Upload a file to Cloudinary"""
    logger.info(f"Uploading to Cloudinary with public ID: {public_id}")
    try:
        result = cloudinary.uploader.upload(
            file_path,
            public_id=public_id,
            resource_type="raw",
            unique_filename=True,
            overwrite=True
        )
        return result["secure_url"]
    except Exception as e:
        logger.error(f"Error uploading to Cloudinary: {str(e)}")
        raise e

def extract_cloudinary_path_from_url(url):
    """Extract the Cloudinary path from a URL"""
    parsed_url = urlparse(url)
    path = parsed_url.path
    
    # Remove version part if present (v1234567890/)
    path_parts = path.split('/')
    if path_parts and path_parts[1].startswith('v') and path_parts[1][1:].isdigit():
        path_parts.pop(1)
    
    # Join the path parts without the leading slash
    cloudinary_path = '/'.join(path_parts[1:])
    
    # Remove file extension
    cloudinary_path = cloudinary_path.rsplit('.', 1)[0]
    
    return cloudinary_path

def generate_image_with_gemini(prompt):
    """Generate an image using Google's Gemini Pro model"""
    logger.info(f"Generating image with Gemini for prompt: {prompt}")
    
    # Get Gemini API key
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        logger.error("Gemini API key is required for image generation")
        return None
    
    try:
        # Configure Gemini
        genai.configure(api_key=gemini_key)
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Generate the image
        logger.info(f"Calling Gemini model with prompt: {prompt}")
        
        try:
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
            
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'image') and part.image:
                        # Convert the image data to PIL Image
                        image_data = BytesIO(part.image.data)
                        return Image.open(image_data)
                
                logger.error("No image found in Gemini response")
                return None
            else:
                logger.error("No candidates in Gemini response")
                return None
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "rate limit" in str(e).lower():
                logger.error(f"Gemini API rate limit exceeded: {str(e)}")
                logger.info("Consider upgrading your Gemini API plan or waiting a few minutes before trying again")
            else:
                logger.error(f"Error in Gemini API call: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Error generating image with Gemini: {str(e)}")
        return None

def generate_image_with_huggingface(prompt):
    """Generate an image using Hugging Face API"""
    logger.info(f"Generating image with Hugging Face API for prompt: {prompt}")
    
    from huggingface_hub import InferenceClient
    
    # Get Hugging Face token
    huggingface_token = os.environ.get("HUGGINGFACE_API_KEY")
    if not huggingface_token:
        logger.error("Hugging Face token is required")
        return None
    
    try:
        # Initialize the Hugging Face Inference client
        client = InferenceClient(token=huggingface_token)
        
        # Model ID from environment
        model_id = os.environ.get("SD_MODEL_ID", "stabilityai/stable-diffusion-3-medium-diffusers")
        logger.info(f"Using Hugging Face model: {model_id}")
        
        # Generate the image
        logger.info(f"Calling model {model_id} with prompt: {prompt}")
        
        # Add additional parameters to improve image quality
        image = client.text_to_image(
            prompt=prompt,
            model=model_id,
            negative_prompt="low quality, bad anatomy, worst quality, low resolution, blurry",
            num_inference_steps=30,  # More steps for better quality
            guidance_scale=7.5,
            width=1024,  # Higher resolution
            height=1024
        )
        
        if image:
            logger.info("Successfully generated image with Hugging Face API")
            return image
        else:
            logger.error("Hugging Face API returned None")
            return None
        
    except Exception as e:
        logger.error(f"Error generating image with Hugging Face API: {str(e)}")
        return None

def generate_image(prompt, use_huggingface=True):
    """Generate an image using available APIs"""
    logger.info(f"Generating image for prompt: {prompt}")
    
    if use_huggingface:
        # Try Hugging Face first (as requested)
        image = generate_image_with_huggingface(prompt)
        if image:
            logger.info("Successfully generated image with Hugging Face API")
            return image
        else:
            logger.warning("Hugging Face API image generation failed, trying Gemini")
    
    # Try Gemini as fallback
    try:
        image = generate_image_with_gemini(prompt)
        if image:
            logger.info("Successfully generated image with Gemini")
            return image
    except Exception as e:
        logger.error(f"Gemini image generation failed: {str(e)}")
        
        # If Hugging Face wasn't already tried, try it now
        if not use_huggingface:
            logger.info("Trying Hugging Face API as fallback")
            image = generate_image_with_huggingface(prompt)
            if image:
                logger.info("Successfully generated image with Hugging Face API fallback")
                return image
    
    logger.error("All image generation methods failed")
    return None

def process_image_to_3d(image_url, prompt=None, use_huggingface=True):
    """Process an image to create a 3D model
    
    Args:
        image_url: URL of the image to process
        prompt: Optional text prompt to generate a new image
        use_huggingface: Whether to use Hugging Face API for image generation (default: True)
    
    Returns:
        dict: Contains URLs for the depth map and 3D model, and a success flag
    """
    logger.info(f"Processing image from {image_url}")
    
    try:
        # If prompt is provided, generate a new image
        image = None
        if prompt:
            logger.info(f"Generating new image for prompt: {prompt}")
            image = generate_image(prompt, use_huggingface=use_huggingface)
            
            if not image:
                logger.error("Image generation failed")
                return {"error": "Failed to generate image", "success": False}
            
            logger.info("Successfully generated image, using it for 3D conversion")
        else:
            # No prompt provided, just download the existing image
            image = download_image(image_url)
        
        # Generate depth map
        depth_map = generate_depth_map(image)
        
        # Create point cloud and then OBJ file
        points, colors = create_point_cloud(image, depth_map)
        obj_file_path = create_obj_file(points, colors)
        
        # Extract Cloudinary path from image URL for consistent naming
        cloudinary_path = extract_cloudinary_path_from_url(image_url)
        timestamp = int(time.time())
        
        # Create public IDs for uploads
        if cloudinary_path.startswith("image_"):
            depth_public_id = f"depth_maps/{cloudinary_path.replace('image_', 'depth_', 1)}"
            model_public_id = f"3d_models/{cloudinary_path.replace('image_', 'model_', 1)}"
        else:
            depth_public_id = f"depth_maps/depth_{timestamp}"
            model_public_id = f"3d_models/model_{timestamp}"
        
        # Save and upload depth map
        depth_map_buffer = io.BytesIO()
        depth_map.save(depth_map_buffer, format='PNG')
        depth_map_buffer.seek(0)
        
        depth_response = cloudinary.uploader.upload(
            depth_map_buffer,
            public_id=depth_public_id,
            resource_type="image",
            unique_filename=True,
            overwrite=True
        )
        
        # Upload OBJ file
        obj_response = cloudinary.uploader.upload(
            obj_file_path,
            public_id=model_public_id,
            resource_type="raw",
            unique_filename=True,
            overwrite=True
        )
        
        # Clean up temporary files
        os.unlink(obj_file_path)
        
        return {
            "depth_map_url": depth_response["secure_url"],
            "model_url": obj_response["secure_url"],
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Failed to process image: {str(e)}")
        return {
            "error": str(e),
            "success": False
        }
