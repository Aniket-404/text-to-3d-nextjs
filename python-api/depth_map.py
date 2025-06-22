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
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-beit-large-512")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-large-512").to(device)
    if device.type == "cuda":
        logger.info(f"Successfully initialized DPT model on GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("GPU not available, initialized DPT model on CPU")
except Exception as e:
    logger.error(f"Failed to initialize DPT model: {str(e)}")
    raise

def download_image(image_url):
    """Download an image from a URL (optimized for Cloudinary URLs)"""
    logger.info(f"Downloading image from {image_url}")
    try:
        # Special handling for Cloudinary URLs to ensure best quality
        if "cloudinary.com" in image_url:
            # Remove any transformation parameters and request original quality
            parsed_url = urlparse(image_url)
            path_parts = parsed_url.path.split('/')
            
            # Rebuild the URL without transformation parameters
            # Format: https://res.cloudinary.com/cloud_name/image/upload/v1234567890/path/to/image.jpg
            filtered_parts = []
            upload_found = False
            for part in path_parts:
                if upload_found or part == "upload":
                    upload_found = True
                    filtered_parts.append(part)
            
            # Reconstruct the URL with original quality parameters
            new_path = '/'.join(filtered_parts)
            clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}/{new_path}?fl_attachment=true"
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
    """Generate a depth map from an image using DPT model"""
    logger.info("Generating depth map using DPT")
    
    try:
        # Get the current device
        device = next(model.parameters()).device
        logger.info(f"Using device: {device}")
        
        # Prepare image for the model
        inputs = processor(images=image, return_tensors="pt")
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
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
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)
        
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
    
    # Loop through each pixel
    for v in range(depth_array.shape[0]):
        for u in range(depth_array.shape[1]):
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
    
    # Initial downsampling to reduce processing time
    logger.info(f"Initial point cloud size: {len(point_cloud)} points")
    if len(point_cloud) > 100000:  # If more than 100k points
        voxel_size = 0.01  # Start with 1cm voxel size
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        logger.info(f"Downsampled to {len(pcd.points)} points using voxel size {voxel_size}")
    
    # Create a mesh from the point cloud
    try:        # Estimate normals with optimized parameters
        logger.info("Starting normal estimation (this may take a few moments)...")
        # Use fewer nearest neighbors and smaller radius for faster computation
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=20)
        )
        logger.info("Basic normal estimation complete, orienting normals...")
        
        # Reduce k parameter for faster orientation
        pcd.orient_normals_consistent_tangent_plane(k=15)
        logger.info("Normal estimation completed")
        
        # Optional: save point cloud visualization
        temp_vis_path = os.path.join("temp", f"pointcloud_vis_{int(time.time())}.png")
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(temp_vis_path)
            vis.destroy_window()
            logger.info(f"Saved point cloud visualization to {temp_vis_path}")
        except Exception as e:
            logger.warning(f"Could not save point cloud visualization: {str(e)}")
        
        # Start with higher quality settings
        depth = 9
        points_percent = 1.0
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            # Downsample point cloud if needed
            working_pcd = pcd
            if points_percent < 1.0:
                n_points = len(np.asarray(pcd.points))
                every_k_points = int(1 / points_percent)
                working_pcd = pcd.uniform_down_sample(every_k_points)
                logger.info(f"Downsampled point cloud to {points_percent*100}% ({len(np.asarray(working_pcd.points))} points)")
            
            # Create mesh using Poisson reconstruction
            logger.info(f"Creating mesh using Poisson reconstruction (depth={depth})...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                working_pcd, 
                depth=depth,
                width=0,
                scale=1.1,
                linear_fit=True
            )
            
            # Clean up the mesh
            logger.info("Cleaning up mesh...")
            # Remove low density vertices (noise)
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
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
            
            # Adjust parameters for next attempt
            attempt += 1
            if attempt < max_attempts:
                if depth > 7:
                    depth -= 1
                    logger.info(f"Reducing Poisson depth to {depth}")
                else:
                    points_percent *= 0.5
                    logger.info(f"Reducing point cloud density to {points_percent*100}%")
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
