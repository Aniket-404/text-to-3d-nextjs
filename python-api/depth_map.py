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
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Load GitHub API key (for authenticated GitHub API requests)
GITHUB_API_KEY = os.environ.get("GITHUB_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
)

# Initialize the DPT model and processor with CUDA if available
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use Intel's DPT model for depth estimation (not FLUX which is for text-to-image)
    model_name = "Intel/dpt-beit-large-512"  # High quality depth model
    
    processor = DPTImageProcessor.from_pretrained(model_name)
    model = DPTForDepthEstimation.from_pretrained(model_name).to(device)
    
    if device.type == "cuda":
        logger.info(f"Successfully initialized DPT model '{model_name}' on GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"GPU not available, initialized DPT model '{model_name}' on CPU")
except Exception as e:
    logger.error(f"Failed to initialize DPT model: {str(e)}")
    # Try fallback to smaller/faster model if the large one fails
    try:
        model_name = "Intel/dpt-swinv2-tiny-256"
        logger.info(f"Trying fallback model: {model_name}")
        processor = DPTImageProcessor.from_pretrained(model_name)
        model = DPTForDepthEstimation.from_pretrained(model_name).to(device)
        logger.info(f"Successfully initialized fallback DPT model '{model_name}'")
    except Exception as fallback_e:
        logger.error(f"Failed to initialize fallback DPT model: {str(fallback_e)}")
        raise

def download_image(image_url):
    """Download an image from a URL (optimized for Cloudinary URLs)"""
    logger.info(f"Downloading image from {image_url}")
    try:
        # Special handling for Cloudinary URLs to ensure best quality
        if "cloudinary.com" in image_url:
            # For Cloudinary URLs, just add quality optimization parameters
            parsed_url = urlparse(image_url)
            # Add fl_attachment=true to ensure we get the original file
            if parsed_url.query:
                clean_url = f"{image_url}&fl_attachment=true"
            else:
                clean_url = f"{image_url}?fl_attachment=true"
            logger.info(f"Optimized Cloudinary URL: {clean_url}")
            response = requests.get(clean_url, timeout=30)
        else:
            # Regular URL download
            response = requests.get(image_url, timeout=30)
        
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            # Log image details for debugging
            logger.info(f"Downloaded image: size={img.size}, mode={img.mode}")
            
            # Ensure image is in RGB mode for processing
            if img.mode != "RGB":
                logger.info(f"Converting image from {img.mode} to RGB mode")
                img = img.convert("RGB")
                
            return img
        else:
            raise ValueError(f"Failed to download image: HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"Error downloading image from {image_url}: {str(e)}")
        raise

def generate_depth_map(image, job_id=None):
    """Generate a depth map from an image using DPT model"""
    logger.info("Generating depth map using DPT")
    
    # Function to check for cancellation
    def check_cancellation():
        if job_id:
            try:
                # Import here to avoid circular imports
                from app import is_job_cancelled
                if is_job_cancelled(job_id):
                    logger.info(f"Job {job_id} was cancelled during depth map generation")
                    return True
            except ImportError:
                # If we can't import, assume not cancelled
                pass
        return False
    
    try:
        # Check for cancellation at the start
        if check_cancellation():
            raise Exception("Job was cancelled during depth map generation")
        
        # Get the current device
        device = next(model.parameters()).device
        logger.info(f"Using device: {device}")
        
        # Check for cancellation before preprocessing
        if check_cancellation():
            raise Exception("Job was cancelled during depth map generation")
        
        # Prepare image for the model
        inputs = processor(images=image, return_tensors="pt")
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Check for cancellation before model inference
        if check_cancellation():
            raise Exception("Job was cancelled during depth map generation")
        
        # Generate depth prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Check for cancellation before post-processing
        if check_cancellation():
            raise Exception("Job was cancelled during depth map generation")

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # Convert to numpy array and normalize
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)
        
        # Check for cancellation before saving
        if check_cancellation():
            raise Exception("Job was cancelled during depth map generation")
        
        # Save a visualization of the depth map and upload to Cloudinary
        timestamp = int(time.time())
        temp_path = os.path.join("temp", f"depth_vis_{timestamp}.png")
        
        try:
            plt.figure(figsize=(10, 10))
            plt.imshow(depth, cmap='gray')
            plt.axis('off')
            plt.savefig(temp_path)
            plt.close()
            logger.info(f"Saved depth map visualization to {temp_path}")
              # Upload depth map visualization to Cloudinary using same name pattern as model
            base_name = image.filename if hasattr(image, 'filename') else f"image_{timestamp}"
            depth_public_id = f"text-to-3d-web/depth-maps/{base_name}_{timestamp}"
            depth_response = cloudinary.uploader.upload(
                temp_path,
                public_id=depth_public_id,
                resource_type="image",
                unique_filename=True,
                overwrite=True,
                quality="auto"
            )
            logger.info(f"Uploaded depth map to Cloudinary: {depth_response['secure_url']}")
            # Store the URL in a property that can be accessed later
            depth.cloudinary_url = depth_response['secure_url']
        except Exception as e:
            logger.warning(f"Could not save or upload depth visualization: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary depth map file: {str(e)}")
        
        return depth
    except Exception as e:
        logger.error(f"Failed to generate depth map: {str(e)}")
        raise

def create_point_cloud(image, depth_map, fx=1000, fy=1000, job_id=None):
    """Create a point cloud from an image and its depth map"""
    logger.info("Creating point cloud")
    
    # Function to check for cancellation
    def check_cancellation():
        if job_id:
            try:
                # Import here to avoid circular imports
                from app import is_job_cancelled
                if is_job_cancelled(job_id):
                    logger.info(f"Job {job_id} was cancelled during point cloud creation")
                    return True
            except ImportError:
                # If we can't import, assume not cancelled
                pass
        return False
    
    # Check for cancellation at the start
    if check_cancellation():
        raise Exception("Job was cancelled during point cloud creation")
    
    # Convert inputs to numpy arrays
    depth_array = np.array(depth_map)
    color_array = np.array(image)
    
    # Calculate camera parameters (image center)
    cx = color_array.shape[1] / 2
    cy = color_array.shape[0] / 2
    
    logger.info(f"Image dimensions: {color_array.shape[1]}x{color_array.shape[0]}")
    logger.info(f"Camera parameters: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # Generate 3D point cloud with colors
    point_cloud = []
    colors = []
    
    total_pixels = depth_array.shape[0] * depth_array.shape[1]
    processed_pixels = 0
    
    # Loop through each pixel
    for v in range(depth_array.shape[0]):
        for u in range(depth_array.shape[1]):
            # Check for cancellation periodically (every 10000 pixels)
            if processed_pixels % 10000 == 0 and check_cancellation():
                raise Exception("Job was cancelled during point cloud creation")
            
            Z = depth_array[v, u]  # Depth value
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            point_cloud.append([X, Y, Z])
            colors.append(color_array[v, u])
            processed_pixels += 1
    
    logger.info(f"Generated point cloud with {len(point_cloud)} points")
    
    # Check for cancellation before final processing
    if check_cancellation():
        raise Exception("Job was cancelled during point cloud creation")
    
    # Convert to numpy arrays and normalize colors
    points = np.array(point_cloud)
    colors = np.array(colors) / 255.0
    
    return points, colors

def create_obj_file(point_cloud, colors, job_id=None):
    """Create an OBJ file from point cloud data, ensuring file size is under 10MB"""
    logger.info("Creating OBJ file")
    
    # Function to check for cancellation
    def check_cancellation():
        if job_id:
            try:
                # Import here to avoid circular imports
                from app import is_job_cancelled
                if is_job_cancelled(job_id):
                    logger.info(f"Job {job_id} was cancelled during OBJ file creation")
                    return True
            except ImportError:
                # If we can't import, assume not cancelled
                pass
        return False
    
    # Check for cancellation at the start
    if check_cancellation():
        raise Exception("Job was cancelled during OBJ file creation")
    
    def check_file_size(filepath):
        """Check if file size is under 10MB"""
        size_mb = os.path.getsize(filepath) / (1024 * 1024)  # Convert to MB
        logger.info(f"Current OBJ file size: {size_mb:.2f}MB")
        return size_mb < 10
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
      # Aggressive initial downsampling for performance
    logger.info(f"Initial point cloud size: {len(point_cloud)} points")
    target_points = 50000  # Target 50k points for good balance
    if len(point_cloud) > target_points:
        # Check for cancellation before downsampling
        if check_cancellation():
            raise Exception("Job was cancelled during OBJ file creation")
        
        # Calculate voxel size to achieve target point count
        # Assuming uniform distribution, cube root of ratio gives approximate voxel size
        ratio = (len(point_cloud) / target_points) ** (1/3)
        voxel_size = 0.02 * ratio  # Base size 2cm * ratio
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        logger.info(f"Downsampled to {len(pcd.points)} points using voxel size {voxel_size:.3f}")
    
    # Check for cancellation before mesh creation
    if check_cancellation():
        raise Exception("Job was cancelled during OBJ file creation")
    
    # Create a mesh from the point cloud
    try:
        # Estimate normals with optimized parameters
        logger.info("Starting normal estimation...")
        
        # Check for cancellation before normal estimation
        if check_cancellation():
            raise Exception("Job was cancelled during OBJ file creation")
        
        # Use optimized parameters for faster normal estimation
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1,  # Larger radius
                max_nn=15    # Fewer neighbors
            )
        )
        logger.info("Basic normal estimation complete, orienting normals...")
        
        # Check for cancellation before normal orientation
        if check_cancellation():
            raise Exception("Job was cancelled during OBJ file creation")
        
        # Faster normal orientation with fewer neighbors
        pcd.orient_normals_consistent_tangent_plane(k=10)
        logger.info("Normal estimation completed")
        
        # Skip point cloud visualization for faster processing
        
        # Start with optimized quality settings
        depth = 8  # Start with depth 8 which usually gives good results
        points_percent = 1.0
        max_attempts = 2  # Reduce max attempts since we're starting with better params
        attempt = 0
        
        while attempt < max_attempts:
            # Check for cancellation before each attempt
            if check_cancellation():
                raise Exception("Job was cancelled during OBJ file creation")
            
            # Downsample point cloud if needed
            working_pcd = pcd
            if points_percent < 1.0:
                n_points = len(np.asarray(pcd.points))
                every_k_points = int(1 / points_percent)
                working_pcd = pcd.uniform_down_sample(every_k_points)
                logger.info(f"Downsampled point cloud to {points_percent*100}% ({len(np.asarray(working_pcd.points))} points)")
              # Create mesh using optimized Poisson reconstruction
            logger.info(f"Creating mesh using Poisson reconstruction (depth={depth})...")
            
            # Check for cancellation before Poisson reconstruction
            if check_cancellation():
                raise Exception("Job was cancelled during OBJ file creation")
            
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                working_pcd, 
                depth=depth,
                width=0,
                scale=1.0,  # Reduced scale for faster processing
                linear_fit=True
            )
            
            # Check for cancellation before mesh cleanup
            if check_cancellation():
                raise Exception("Job was cancelled during OBJ file creation")
            
            # Clean up the mesh
            logger.info("Cleaning up mesh...")
            # More aggressive noise removal
            vertices_to_remove = densities < np.quantile(densities, 0.15)  # Remove more low-density vertices
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            # Check for cancellation before mesh optimization
            if check_cancellation():
                raise Exception("Job was cancelled during OBJ file creation")
            
            # Optimize the mesh
            if len(np.asarray(mesh.vertices)) > target_points:
                # Decimate mesh to reduce complexity while preserving shape
                reduction_ratio = target_points / len(np.asarray(mesh.vertices))
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(len(np.asarray(mesh.triangles)) * reduction_ratio))
            
            # Optimize mesh
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
            # Check for cancellation before writing file
            if check_cancellation():
                raise Exception("Job was cancelled during OBJ file creation")
            
            # Create temporary file for the mesh in the project's temp directory
            temp_obj_path = os.path.join("temp", f"mesh_{int(time.time())}.obj")
            logger.info(f"Writing mesh to {temp_obj_path}")
            o3d.io.write_triangle_mesh(
                temp_obj_path,
                mesh,
                write_vertex_normals=True,
                write_vertex_colors=True
            )
            
            # Check file size
            if check_file_size(temp_obj_path):
                logger.info("OBJ file size is within limits")
                return temp_obj_path
            
            # Clean up the temporary file if too large
            os.remove(temp_obj_path)
              # Adjust parameters for next attempt
            attempt += 1
            if attempt < max_attempts:
                # On second attempt, reduce both depth and point density
                depth = 6
                points_percent = 0.5
                logger.info(f"Reducing parameters: depth={depth}, density={points_percent*100}%")
                continue
            
        logger.error("Failed to create OBJ file under 10MB after multiple attempts")
        raise ValueError("Could not generate OBJ file under 10MB size limit")
            
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
        model = genai.GenerativeModel('gemini-2.5-flash')
        
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

def generate_image_with_github(prompt):
    """Generate an image using GitHub's OpenAI proxy API (if supported)"""
    logger.info(f"Generating image with GitHub OpenAI proxy for prompt: {prompt}")
    try:
        from openai import OpenAI
        token = os.environ.get("GITHUB_API_KEY") or os.environ.get("GITHUB_TOKEN")
        if not token:
            logger.error("GitHub API key is required for GitHub image generation")
            return None
        endpoint = "https://models.github.ai/inference"
        model_name = os.environ.get("GITHUB_IMAGE_MODEL", "openai/dall-e-3")
        client = OpenAI(
            base_url=endpoint,
            api_key=token,
        )
        logger.info(f"Calling GitHub OpenAI proxy with model: {model_name}")
        try:
            response = client.images.generate(
                model=model_name,
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            if response and response.data and response.data[0].url:
                image_url = response.data[0].url
                logger.info(f"Image generated, downloading from {image_url}")
                img = download_image(image_url)
                return img
            else:
                logger.error("No image URL returned from GitHub OpenAI proxy")
                return None
        except Exception as e:
            if "404" in str(e):
                logger.error("GitHub OpenAI proxy does not support image generation (404 Not Found). Skipping this fallback.")
            else:
                logger.error(f"Error generating image with GitHub OpenAI proxy: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error generating image with GitHub OpenAI proxy: {str(e)}")
        return None

def generate_image(prompt, use_huggingface=True, use_github=False):
    """Generate an image using available APIs, optionally using GitHub proxy"""
    logger.info(f"Generating image for prompt: {prompt}")
    if use_github:
        image = generate_image_with_github(prompt)
        if image:
            logger.info("Successfully generated image with GitHub OpenAI proxy")
            return image
        else:
            logger.warning("GitHub OpenAI proxy image generation failed, trying Hugging Face/Gemini")
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
        if not use_huggingface and not use_github:
            logger.info("Trying Hugging Face API as fallback")
            image = generate_image_with_huggingface(prompt)
            if image:
                logger.info("Successfully generated image with Hugging Face API fallback")
                return image
    
    logger.error("All image generation methods failed")
    return None

def process_image_to_3d(image_url, prompt=None, use_huggingface=True, job_id=None):
    """Process an image to create a 3D model
    
    Args:
        image_url: URL of the image to process (Cloudinary URL)
        prompt: Optional text prompt to generate a new image
        use_huggingface: Whether to use Hugging Face API for image generation (default: True)
        job_id: Optional job ID to check for cancellation
    
    Returns:
        dict: Contains URLs for the image and 3D model, and a success flag
    """
    logger.info(f"Processing image to 3D from URL: {image_url} (Job ID: {job_id})")
    
    # Function to check for cancellation
    def check_cancellation():
        if job_id:
            try:
                # Import here to avoid circular imports
                from app import is_job_cancelled
                if is_job_cancelled(job_id):
                    logger.info(f"Job {job_id} was cancelled, stopping processing")
                    return True
            except ImportError:
                # If we can't import, assume not cancelled
                pass
        return False
    
    # Function to update progress
    def update_progress(stage, progress, message):
        if job_id:
            try:
                from app import update_job_progress
                update_job_progress(job_id, stage, progress, message)
            except ImportError:
                pass
    
    try:
        # Check for cancellation at the start
        if check_cancellation():
            return {"error": "Job was cancelled", "success": False, "cancelled": True}
        
        # If prompt is provided, generate a new image
        image = None
        generated_image_url = None
        
        if prompt:
            logger.info(f"Generating new image for prompt: {prompt}")
            update_progress('generating_image', 10, 'Generating image from text prompt...')
            
            # Check for cancellation before image generation
            if check_cancellation():
                return {"error": "Job was cancelled", "success": False, "cancelled": True}
            
            image = generate_image(prompt, use_huggingface=use_huggingface)
            if not image:
                logger.warning("Hugging Face/Gemini image generation failed, trying GitHub OpenAI proxy...")
                update_progress('generating_image', 15, 'Retrying image generation with fallback...')
                
                # Check for cancellation before retry
                if check_cancellation():
                    return {"error": "Job was cancelled", "success": False, "cancelled": True}
                
                image = generate_image(prompt, use_huggingface=False, use_github=True)
            if not image:
                logger.error("Image generation failed")
                return {"error": "Failed to generate image", "success": False}
            
            logger.info("Successfully generated image, uploading to Cloudinary")
            update_progress('uploading_image', 25, 'Uploading generated image to cloud...')
            
            # Check for cancellation before uploading
            if check_cancellation():
                return {"error": "Job was cancelled", "success": False, "cancelled": True}
            
            # Upload the generated image to Cloudinary
            timestamp = int(time.time())
            safe_prompt = ''.join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
            
            # Save image to a temporary file
            temp_img_path = os.path.join("temp", f"generated_{timestamp}.png")
            image.save(temp_img_path)            # Upload to Cloudinary in the text-to-3d-web/images folder
            img_public_id = f"text-to-3d-web/images/{safe_prompt}_{timestamp}"
            img_response = cloudinary.uploader.upload(
                temp_img_path,
                public_id=img_public_id,
                resource_type="image",
                unique_filename=True,
                overwrite=True,
                quality="auto",
                fetch_format="auto",
                eager_async=True,
                eager=[
                    {"width": 1024, "height": 1024, "crop": "fill"},
                    {"width": 512, "height": 512, "crop": "fill"}
                ]
            )
            
            # Clean up temporary file
            os.remove(temp_img_path)
            
            # Use the new Cloudinary URL for further processing
            generated_image_url = img_response["secure_url"]
            image_url = generated_image_url
            
            logger.info(f"Uploaded generated image to Cloudinary: {generated_image_url}")
        else:
            logger.info("No prompt provided, using the original image URL")
        
        # Check for cancellation before downloading
        if check_cancellation():
            return {"error": "Job was cancelled", "success": False, "cancelled": True}
        
        # Download image from Cloudinary if we don't already have it
        if not image:
            update_progress('downloading_image', 35, 'Downloading image for processing...')
            image = download_image(image_url)
          # Get base name for consistent file naming
        base_path = None
        if prompt:
            base_path = safe_prompt
        elif image_url:
            cloudinary_path = extract_cloudinary_path_from_url(image_url)
            base_path = cloudinary_path.split('/')[-1]
        if not base_path:
            base_path = f"image_{int(time.time())}"
            
        # Check for cancellation before depth map generation
        if check_cancellation():
            return {"error": "Job was cancelled", "success": False, "cancelled": True}
            
        # Generate depth map and create point cloud
        logger.info("Generating depth map...")
        update_progress('generating_depth', 40, 'Generating depth map from image...')
        # Set filename attribute for consistent naming
        image.filename = base_path
        depth_map = generate_depth_map(image, job_id=job_id)
        depth_map_url = getattr(depth_map, 'cloudinary_url', None)  # Get the Cloudinary URL if available
        
        # Check for cancellation before point cloud creation
        if check_cancellation():
            return {"error": "Job was cancelled", "success": False, "cancelled": True}
        
        logger.info("Creating point cloud...")
        update_progress('creating_pointcloud', 60, 'Creating 3D point cloud...')
        points, colors = create_point_cloud(image, depth_map, job_id=job_id)
        
        # Check for cancellation before OBJ file creation
        if check_cancellation():
            return {"error": "Job was cancelled", "success": False, "cancelled": True}
        
        logger.info("Creating OBJ file...")
        update_progress('creating_mesh', 75, 'Creating 3D mesh and optimizing...')
        obj_file_path = create_obj_file(points, colors, job_id=job_id)
        
        # Extract Cloudinary path from image URL for consistent naming
        cloudinary_path = extract_cloudinary_path_from_url(image_url)
        timestamp = int(time.time())
        base_path = cloudinary_path.split('/')[-1]          # Create public IDs for uploads using the text-to-3d-web/models folder
        model_public_id = f"text-to-3d-web/models/{base_path}_{timestamp}"
        
        # Check for cancellation before uploading model
        if check_cancellation():
            # Clean up temporary file before returning
            if os.path.exists(obj_file_path):
                os.unlink(obj_file_path)
            return {"error": "Job was cancelled", "success": False, "cancelled": True}
        
        # Upload OBJ file to models folder
        logger.info(f"Uploading 3D model to Cloudinary as {model_public_id}")
        update_progress('uploading_model', 90, 'Uploading 3D model to cloud...')
        obj_response = cloudinary.uploader.upload(
            obj_file_path,
            public_id=model_public_id,
            resource_type="raw",
            unique_filename=True,
            overwrite=True
        )
        
        # Clean up temporary files
        os.unlink(obj_file_path)
        
        update_progress('completed', 100, 'Generation completed successfully!')
        
        # Prepare result with all URLs
        result = {
            "model_url": obj_response["secure_url"],
            "success": True
        }
        
        # Add generated image URL if we created one
        if generated_image_url:
            result["generated_image_url"] = generated_image_url
            
        # Add depth map URL if available
        if depth_map_url:
            result["depth_map_url"] = depth_map_url
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to process image to 3D: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "success": False
        }

def github_api_request(endpoint, method="GET", params=None, data=None, headers=None, stream=False):
    """
    Make an authenticated request to the GitHub API using the GITHUB_API_KEY.
    Args:
        endpoint (str): The GitHub API endpoint (e.g., '/repos/owner/repo/contents/path/to/file').
        method (str): HTTP method ('GET', 'POST', etc.).
        params (dict): Query parameters.
        data (dict): Data for POST/PUT requests.
        headers (dict): Additional headers.
        stream (bool): Whether to stream the response.
    Returns:
        Response object or None if error.
    """
    if not GITHUB_API_KEY:
        logger.error("GITHUB_API_KEY is not set in environment variables.")
        return None
    base_url = "https://api.github.com"
    url = base_url + endpoint
    req_headers = {
        "Authorization": f"token {GITHUB_API_KEY}",
        "Accept": "application/vnd.github+json"
    }
    if headers:
        req_headers.update(headers)
    try:
        response = requests.request(method, url, params=params, json=data, headers=req_headers, stream=stream, timeout=30)
        response.raise_for_status()
        return response
    except Exception as e:
        logger.error(f"GitHub API request failed: {str(e)}")
        return None

# Example: Fetch a file from a GitHub repo (returns raw content as bytes)
def fetch_github_file(owner, repo, path, ref="main"):
    """
    Fetch a file from a GitHub repository using the API key.
    Args:
        owner (str): Repository owner.
        repo (str): Repository name.
        path (str): File path in the repo.
        ref (str): Branch or commit SHA (default: 'main').
    Returns:
        bytes: File content, or None if error.
    """
    endpoint = f"/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": ref}
    response = github_api_request(endpoint, params=params)
    if response and response.status_code == 200:
        content_json = response.json()
        if content_json.get("encoding") == "base64":
            import base64
            return base64.b64decode(content_json["content"])
        else:
            logger.error("Unexpected encoding in GitHub file response.")
    else:
        logger.error(f"Failed to fetch file from GitHub: {owner}/{repo}/{path}")
    return None
