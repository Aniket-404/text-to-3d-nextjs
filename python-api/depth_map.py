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
import torch
import torch.nn as nn
from transformers import DPTImageProcessor, DPTForDepthEstimation
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

# Initialize the DPT model for depth estimation
try:
    # Check if GPU is available for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"PyTorch device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load the DPT Large BEiT 512 model from the local safetensors file
    model_path = os.path.join(os.path.dirname(__file__), "models", "dpt-beit-large-512")
    logger.info(f"Loading DPT Large BEiT 512 model from: {model_path}")
    
    # Load the model and processor
    processor = DPTImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = DPTForDepthEstimation.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    model.eval()
    
    logger.info("Successfully loaded DPT Large BEiT 512 model for depth estimation")
    
    # Test GPU functionality if available
    if torch.cuda.is_available():
        try:
            test_tensor = torch.tensor([[1.0]], device=device)
            result = torch.square(test_tensor)
            logger.info(f"PyTorch GPU test successful: {result.device}")
        except Exception as e:
            logger.warning(f"GPU test failed: {e}, falling back to CPU")
            device = torch.device("cpu")
            model.to(device)
        
except Exception as e:
    logger.error(f"Failed to initialize DPT depth model: {str(e)}")
    raise

def download_image(image_url):
    """Download an image from a URL (optimized for Cloudinary URLs)"""
    logger.info(f"Downloading image from {image_url}")
    try:
        # Special handling for Cloudinary URLs to ensure best quality
        if "cloudinary.com" in image_url:
            # For Cloudinary URLs, just add quality optimization without changing the core URL
            # This preserves the cloud name and proper path structure
            if "?" in image_url:
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

def generate_depth_map(image):
    """Generate a depth map from an image using DPT Large BEiT 512 model with safetensors"""
    logger.info("Generating depth map using DPT Large BEiT 512 model with safetensors")
    
    try:
        # Preprocess the image using the DPT processor
        inputs = processor(images=image, return_tensors="pt")
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logger.info(f"Input tensor shape: {inputs['pixel_values'].shape}")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],  # PIL image size is (width, height), we need (height, width)
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert to numpy and normalize
        depth_array = prediction.squeeze().cpu().numpy()
        
        # Normalize depth values to 0-255 range
        depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        # Convert to PIL Image
        depth = Image.fromarray(depth_uint8)
        
        # Save a visualization of the depth map and upload to Cloudinary
        timestamp = int(time.time())
        temp_path = os.path.join("temp", f"depth_vis_{timestamp}.png")
        
        try:
            # Use a non-GUI backend to avoid threading issues
            import matplotlib
            matplotlib.use('Agg')  # Set non-GUI backend before any matplotlib operations
            
            plt.figure(figsize=(8, 8))  # Keep original size
            plt.imshow(depth, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(temp_path, dpi=100, bbox_inches='tight')
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

def create_point_cloud(image, depth_map, fx=1000, fy=1000):
    """Create a point cloud from an image and its depth map"""
    logger.info("Creating point cloud")
    
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
    
    # For large images, subsample aggressively to reduce processing time
    step = 1
    if depth_array.shape[0] * depth_array.shape[1] > 500000:  # If more than 500k pixels
        step = 4  # Sample every 4th pixel for very large images
        logger.info(f"Very large image detected ({depth_array.shape[0]}x{depth_array.shape[1]}), subsampling by factor {step}")
    elif depth_array.shape[0] * depth_array.shape[1] > 200000:  # If more than 200k pixels
        step = 3  # Sample every 3rd pixel
        logger.info(f"Large image detected ({depth_array.shape[0]}x{depth_array.shape[1]}), subsampling by factor {step}")
    elif depth_array.shape[0] * depth_array.shape[1] > 100000:  # If more than 100k pixels
        step = 2  # Sample every 2nd pixel
        logger.info(f"Medium image detected ({depth_array.shape[0]}x{depth_array.shape[1]}), subsampling by factor {step}")
    
    # Loop through each pixel with subsampling
    for v in range(0, depth_array.shape[0], step):
        for u in range(0, depth_array.shape[1], step):
            Z = depth_array[v, u]  # Depth value
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            point_cloud.append([X, Y, Z])
            colors.append(color_array[v, u])
    
    logger.info(f"Generated point cloud with {len(point_cloud)} points")
    
    # Convert to numpy arrays and normalize colors
    points = np.array(point_cloud)
    colors = np.array(colors) / 255.0
    
    return points, colors

def create_obj_file(point_cloud, colors):
    """Create an OBJ file from point cloud data, ensuring file size is under 10MB"""
    logger.info("Creating OBJ file")
    
    def check_file_size(filepath):
        """Check if file size is under 10MB"""
        size_mb = os.path.getsize(filepath) / (1024 * 1024)  # Convert to MB
        logger.info(f"Current OBJ file size: {size_mb:.2f}MB")
        return size_mb < 10
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
      # Smart downsampling for performance
    logger.info(f"Initial point cloud size: {len(point_cloud)} points")
    
    # Target around 50k points for better quality while maintaining speed
    target_points = 50000
    if len(point_cloud) > target_points:
        # Calculate voxel size to achieve target point count
        # Use moderate downsampling with reasonable voxel size
        ratio = (len(point_cloud) / target_points) ** (1/3)
        voxel_size = 0.05 * ratio  # Balanced base size for quality vs speed
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        logger.info(f"Downsampled to {len(pcd.points)} points using voxel size {voxel_size:.3f}")
    
    # Only use uniform downsampling if we still have too many points (very large datasets)
    if len(pcd.points) > target_points * 1.5:  # Allow some flexibility
        # Use uniform downsampling as a fallback
        downsample_factor = max(1, int(len(pcd.points) / target_points))
        pcd = pcd.uniform_down_sample(every_k_points=downsample_factor)
        logger.info(f"Further downsampled to {len(pcd.points)} points using uniform sampling (every {downsample_factor} points)")
    
    # Create a mesh from the point cloud
    try:
        
        # Estimate normals with optimized parameters
        logger.info("Starting normal estimation...")
        # Use optimized parameters for faster normal estimation
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.15,  # Larger radius for faster computation
                max_nn=10     # Fewer neighbors for speed
            )
        )
        logger.info("Basic normal estimation complete, orienting normals...")
        
        # Skip normal orientation for very large point clouds to save time
        if len(pcd.points) > 75000:  # Adjusted threshold for the new target
            logger.info("Skipping normal orientation for large point cloud to save time")
        else:
            # Faster normal orientation with fewer neighbors
            pcd.orient_normals_consistent_tangent_plane(k=5)  # Restored k to 5 for better quality
            logger.info("Normal orientation completed")
        
        # Start with balanced quality settings
        depth = 7  # Restored to 7 for better quality
        points_percent = 1.0  # Start with 100% of points
        max_attempts = 2  # Keep at 2 attempts
        attempt = 0
        
        while attempt < max_attempts:
            # Downsample point cloud if needed
            working_pcd = pcd
            if points_percent < 1.0:
                n_points = len(np.asarray(pcd.points))
                every_k_points = int(1 / points_percent)
                working_pcd = pcd.uniform_down_sample(every_k_points)
                logger.info(f"Downsampled point cloud to {points_percent*100}% ({len(np.asarray(working_pcd.points))} points)")
              # Create mesh using optimized Poisson reconstruction
            logger.info(f"Creating mesh using Poisson reconstruction (depth={depth})...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                working_pcd, 
                depth=depth,
                width=0,
                scale=1.0,  # Reduced scale for faster processing
                linear_fit=True
            )
            
            # Clean up the mesh
            logger.info("Cleaning up mesh...")
            # More aggressive noise removal
            vertices_to_remove = densities < np.quantile(densities, 0.15)  # Remove more low-density vertices
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            # Optimize the mesh - balanced approach
            if len(np.asarray(mesh.vertices)) > 30000:  # Reasonable threshold
                # Moderate decimation to reduce complexity while preserving quality
                target_triangles = min(40000, len(np.asarray(mesh.triangles)) // 2)  # Less aggressive reduction
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
                logger.info(f"Decimated mesh to {len(np.asarray(mesh.vertices))} vertices, {len(np.asarray(mesh.triangles))} triangles")
            
            # Optimize mesh
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
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
              # Adjust parameters for next attempt - moderate reduction
            attempt += 1
            if attempt < max_attempts:
                # On second attempt, reduce depth and point density moderately
                depth = 6  # Reduce from 7 to 6
                points_percent = 0.7  # Use 70% of points (was 0.3)
                logger.info(f"Reducing parameters for retry: depth={depth}, density={points_percent*100}%")
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

def process_image_to_3d(image_url, prompt=None, use_huggingface=True):
    """Process an image to create a 3D model
    
    Args:
        image_url: URL of the image to process (Cloudinary URL)
        prompt: Optional text prompt to generate a new image
        use_huggingface: Whether to use Hugging Face API for image generation (default: True)
    
    Returns:
        dict: Contains URLs for the image and 3D model, and a success flag
    """
    logger.info(f"Processing image to 3D from URL: {image_url}")
    
    try:
        # If prompt is provided, generate a new image
        image = None
        generated_image_url = None
        
        if prompt:
            logger.info(f"Generating new image for prompt: {prompt}")
            image = generate_image(prompt, use_huggingface=use_huggingface)
            if not image:
                logger.warning("Hugging Face/Gemini image generation failed, trying GitHub OpenAI proxy...")
                image = generate_image(prompt, use_huggingface=False, use_github=True)
            if not image:
                logger.error("Image generation failed")
                return {"error": "Failed to generate image", "success": False}
            
            logger.info("Successfully generated image, uploading to Cloudinary")
            
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
        
        # Download image from Cloudinary if we don't already have it
        if not image:
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
            
        # Generate depth map and create point cloud
        logger.info("Generating depth map...")
        # Set filename attribute for consistent naming
        image.filename = base_path
        depth_map = generate_depth_map(image)
        depth_map_url = getattr(depth_map, 'cloudinary_url', None)  # Get the Cloudinary URL if available
        
        logger.info("Creating point cloud...")
        points, colors = create_point_cloud(image, depth_map)
        logger.info("Creating OBJ file...")
        obj_file_path = create_obj_file(points, colors)
        
        # Extract Cloudinary path from image URL for consistent naming
        cloudinary_path = extract_cloudinary_path_from_url(image_url)
        timestamp = int(time.time())
        base_path = cloudinary_path.split('/')[-1]          # Create public IDs for uploads using the text-to-3d-web/models folder
        model_public_id = f"text-to-3d-web/models/{base_path}_{timestamp}"
        
        # Upload OBJ file to models folder
        logger.info(f"Uploading 3D model to Cloudinary as {model_public_id}")
        obj_response = cloudinary.uploader.upload(
            obj_file_path,
            public_id=model_public_id,
            resource_type="raw",
            unique_filename=True,
            overwrite=True
        )
        
        # Clean up temporary files
        os.unlink(obj_file_path)
        
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
