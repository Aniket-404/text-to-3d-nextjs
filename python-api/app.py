from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import base64
from PIL import Image, ImageDraw
import io
import numpy as np
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import time
import tempfile
import torch
from diffusers import DiffusionPipeline, __version__ as diffusers_version
from huggingface_hub import login, whoami, hf_hub_download
from pathlib import Path
import filelock
import threading
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_with_progress(repo_id, local_dir):
    """Download model files with progress tracking."""
    logger.info(f"Starting download of model files from {repo_id}")
    
    try:
        # First, get the model weights file
        logger.info("Downloading model weights...")
        # Try different weight file names based on the model
        weight_filenames = [
            "model.safetensors",
            "diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors"
        ]
        
        downloaded_main_weights = False
        for filename in weight_filenames:
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=local_dir,
                    resume_download=True,
                    force_download=False
                )
                logger.info(f"Model weights downloaded to {path}")
                downloaded_main_weights = True
                break
            except Exception as e:
                logger.warning(f"Could not download {filename}: {str(e)}")
        
        if not downloaded_main_weights:
            logger.error("Failed to download any model weight files")
            return False
        
        # Then get the other necessary files
        logger.info("Downloading model configuration files...")
        config_files = [
            "config.json",
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt"
        ]
        
        for filename in config_files:
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=local_dir,
                    resume_download=True,
                    force_download=False
                )
                logger.info(f"Downloaded {filename} to {path}")
            except Exception as e:
                logger.warning(f"Failed to download {filename}: {str(e)}")
        
        logger.info("Model download completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}", exc_info=True)
        return False

# Load environment variables
load_dotenv()

# Set up cache directory in our project folder
cache_dir = Path(__file__).parent / "model_cache"
cache_dir.mkdir(exist_ok=True)
os.environ['HF_HOME'] = str(cache_dir)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir / "models")

# Create a lock file directory
lock_dir = cache_dir / "locks"
lock_dir.mkdir(exist_ok=True)
model_lock = filelock.FileLock(str(lock_dir / "model.lock"))
pipeline_lock = threading.Lock()

app = Flask(__name__)
CORS(app)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
)

# Log in to Hugging Face Hub
huggingface_token = os.environ.get("HUGGINGFACE_API_KEY")
if not huggingface_token:
    raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")

try:
    # Verify token and get user info
    user_info = whoami()
    logger.info(f"Successfully logged in to Hugging Face Hub as {user_info['name']}")
    login(token=huggingface_token)
except Exception as e:
    logger.error(f"Failed to log in to Hugging Face Hub: {str(e)}")
    raise

# Initialize SDXL pipeline globally
pipe = None

def get_pipeline():
    global pipe
    
    # Use a thread lock to prevent multiple pipeline initializations
    with pipeline_lock:
        if pipe is None:
            logger.info("Initializing Stable Diffusion pipeline...")
            try:
                # Log versions and system info
                logger.info(f"PyTorch version: {torch.__version__}")
                logger.info(f"Diffusers version: {diffusers_version}")
                logger.info(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                  # Get model ID from environment variables
                model_id = os.environ.get("SD_MODEL_ID", "CompVis/stable-diffusion-v1-4")
                
                logger.info(f"Loading model {model_id}...")
                # Use file lock to prevent concurrent downloads
                with model_lock:
                    # Check if model is already downloaded
                    model_path = cache_dir / "models" / model_id
                    if model_path.exists():
                        logger.info("Model already downloaded, loading from cache...")
                    else:
                        logger.info("Downloading model (this may take a while)...")
                        success = download_with_progress(model_id, str(cache_dir / "models"))
                        if not success:
                            raise RuntimeError("Failed to download model files")
                    
                    # Load the model with optimizations
                    pipe = DiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.float32,  # Use float32 for CPU
                        use_safetensors=True,
                        cache_dir=str(cache_dir / "models"),
                        local_files_only=True  # Only use local files after download
                    )
                
                logger.info("Model loaded, configuring device settings...")
                # Move to GPU if available
                if torch.cuda.is_available():
                    pipe.to("cuda")
                    # Enable memory optimizations
                    pipe.enable_attention_slicing()
                    if torch.__version__.startswith('1.'):
                        pipe.enable_xformers_memory_efficient_attention()
                    else:
                        # Use torch.compile for torch >= 2.0
                        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
                else:
                    logger.info("No GPU available, using CPU with basic optimizations...")
                    # Enable basic optimizations for CPU
                    pipe.enable_attention_slicing(1)
                
                logger.info("Pipeline initialization complete!")
            except Exception as e:
                logger.error(f"Error initializing pipeline: {str(e)}", exc_info=True)
                raise
    
    return pipe

# Temp directory for file storage
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "API is running",
        "config": {
            "model": os.environ.get("SD_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"),
            "cloudinary_configured": all([
                os.environ.get("CLOUDINARY_CLOUD_NAME"),
                os.environ.get("CLOUDINARY_API_KEY"),
                os.environ.get("CLOUDINARY_API_SECRET")
            ]),
            "huggingface_configured": bool(os.environ.get("HUGGINGFACE_API_KEY")),
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    })

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        logger.info(f"Processing prompt: {prompt}")
        
        try:
            # Get the pipeline (initialized on first use)
            pipeline = get_pipeline()
            if pipeline is None:
                raise RuntimeError("Failed to initialize pipeline")
            
            logger.info("Starting image generation...")            # Generate image with smaller model
            image = pipeline(
                prompt=prompt,
                num_inference_steps=20,  # Fewer steps for faster generation
                guidance_scale=7.5,
                width=512,  # Smaller image
                height=512  # Smaller image
            ).images[0]
            logger.info("Image generation complete!")
            
            image_source = "generated"
            
        except Exception as e:
            error_msg = f"Image generation error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            logger.info("Using fallback placeholder image...")
            
            # Use a fallback image
            placeholder_path = os.path.join(os.path.dirname(__file__), "placeholder.png")
            if os.path.exists(placeholder_path):
                image = Image.open(placeholder_path)
            else:
                # Create a simple placeholder image
                image = Image.new('RGB', (1024, 1024), color=(73, 109, 137))
                d = ImageDraw.Draw(image)
                d.text((10, 10), f"Error: {str(e)[:200]}", fill=(255, 255, 0))
                d.text((10, 30), f"Prompt: {prompt[:200]}", fill=(255, 255, 0))
                image.save(placeholder_path)
            
            image_source = "fallback"
        
        logger.info("Processing image for upload...")
        # Save image to a temporary file
        timestamp = int(time.time())
        temp_filename = f"temp_{timestamp}.png"
        temp_path = os.path.join(TEMP_DIR, temp_filename)
        image.save(temp_path)
        
        # Upload to Cloudinary with a unique filename based on the prompt
        safe_filename = ''.join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_filename = safe_filename[:50]  # Truncate to a reasonable length
        
        cloudinary_folder = "text-to-3d-web/images"
        unique_filename = f"{safe_filename}_{timestamp}"
        
        logger.info(f"Uploading to Cloudinary as {unique_filename}...")
        upload_result = cloudinary.uploader.upload(
            temp_path,
            public_id=f"{cloudinary_folder}/{unique_filename}",
            unique_filename=True,
            overwrite=True
        )
        
        # Clean up temporary file
        os.remove(temp_path)
        
        # Return the Cloudinary URL and image source info
        return jsonify({
            "url": upload_result["secure_url"],
            "source": image_source,
            "prompt": prompt,
            "timestamp": timestamp
        })
        
    except Exception as e:
        error_msg = f"Request processing error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            "error": error_msg,
            "details": str(e)
        }), 500

@app.route('/delete', methods=['POST'])
def delete_file():
    data = request.json
    public_id = data.get('publicId')
    
    if not public_id:
        return jsonify({"error": "No public_id provided"}), 400
        
    try:
        # Delete the file from Cloudinary
        result = cloudinary.uploader.destroy(public_id)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": f"Failed to delete file: {str(e)}"}), 500

if __name__ == '__main__':
    # Run without reloader to prevent interrupting model downloads
    app.run(port=5000, debug=True, use_reloader=False)
