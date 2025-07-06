import os
import json
import time
import base64
import io
from PIL import Image, ImageDraw
import cloudinary
import cloudinary.uploader
import logging
import torch
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from depth_map import process_image_to_3d, generate_image

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure directories
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    # Check both PyTorch and TensorFlow GPU availability
    pytorch_gpu = torch.cuda.is_available()
    tensorflow_gpu = len(tf.config.experimental.list_physical_devices('GPU')) > 0
    
    gpu_info = {
        "pytorch_gpu": pytorch_gpu,
        "tensorflow_gpu": tensorflow_gpu,
        "gpu_available": pytorch_gpu or tensorflow_gpu,
        "gpu_name": torch.cuda.get_device_name(0) if pytorch_gpu else "Not available"
    }
    
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
            **gpu_info
        }
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate an image from a text prompt and create its 3D model"""
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    start_time = time.time()
    
    try:
        logger.info(f"Processing prompt: {prompt}")
        logger.info("Starting image generation and 3D processing...")
        
        # Generate image and process to 3D using the unified function from depth_map.py
        result = process_image_to_3d(None, prompt=prompt, use_huggingface=True)
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        if not result["success"]:
            logger.error(f"Failed to process image: {result.get('error')}")
            return jsonify({
                "error": result.get('error', "Failed to process image"),
                "success": False,
                "processing_time": total_time
            }), 500
        
        # Return URLs for both the generated image and 3D model
        return jsonify({
            "image_url": result["generated_image_url"],  # URL of the generated image
            "model_url": result["model_url"],           # URL of the 3D model file
            "depth_map_url": result["depth_map_url"],   # URL of the depth map (optional)
            "success": True,
            "processing_time": total_time
        })
            
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Failed to process request after {total_time:.2f} seconds: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "success": False,
            "processing_time": total_time
        }), 500

@app.route('/delete', methods=['POST'])
def delete_file():
    data = request.json
    public_id = data.get('publicId')
    
    if not public_id:
        return jsonify({"error": "No public_id provided"}), 400
        
    try:
        result = cloudinary.uploader.destroy(public_id)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": f"Failed to delete file: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def upload():
    """Upload an image directly and create its 3D model"""
    try:
        data = request.json
        image_data = data.get('image')
        filename = data.get('filename', 'uploaded_image.jpg')
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        logger.info(f"Processing uploaded image: {filename}")
        
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            logger.info(f"Successfully decoded image: {image.size}")
            
        except Exception as e:
            logger.error(f"Failed to decode image: {str(e)}")
            return jsonify({"error": "Invalid image data"}), 400
        
        start_time = time.time()
        
        # Upload the user's image to Cloudinary first
        logger.info("Uploading user image to Cloudinary...")
        timestamp = int(time.time())
        safe_filename = ''.join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).strip()[:50]
        
        # Save image to a temporary file
        temp_img_path = os.path.join("temp", f"uploaded_{timestamp}.png")
        image.save(temp_img_path)
        
        # Upload to Cloudinary in the text-to-3d-web/images folder
        img_public_id = f"text-to-3d-web/images/uploaded_{safe_filename}_{timestamp}"
        img_response = cloudinary.uploader.upload(
            temp_img_path,
            public_id=img_public_id,
            resource_type="image",
            unique_filename=True,
            overwrite=True,
            quality="auto",
            fetch_format="auto"
        )
        
        # Clean up temporary file
        os.remove(temp_img_path)
        
        uploaded_image_url = img_response["secure_url"]
        logger.info(f"Uploaded user image to Cloudinary: {uploaded_image_url}")
        
        # Process the uploaded image to 3D (without calling Hugging Face API)
        logger.info("Starting 3D processing for uploaded image...")
        result = process_image_to_3d(uploaded_image_url, prompt=None, use_huggingface=False)
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        if not result["success"]:
            logger.error(f"Failed to process uploaded image: {result.get('error')}")
            return jsonify({
                "error": result.get('error', "Failed to process uploaded image"),
                "success": False,
                "processing_time": total_time
            }), 500
        
        # Return URLs for the uploaded image and 3D model
        return jsonify({
            "image_url": uploaded_image_url,              # URL of the uploaded image
            "model_url": result["model_url"],             # URL of the 3D model file
            "depth_map_url": result.get("depth_map_url"), # URL of the depth map
            "success": True,
            "processing_time": total_time,
            "source": "upload"
        })
            
    except Exception as e:
        total_time = time.time() - start_time if 'start_time' in locals() else 0
        logger.error(f"Failed to process uploaded image after {total_time:.2f} seconds: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "success": False,
            "processing_time": total_time
        }), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    """Upload an image and process it to 3D model"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode base64 image
        image_data = data['image']
        filename = data.get('filename', 'uploaded_image.jpg')
        mimetype = data.get('mimetype', 'image/jpeg')
        
        logger.info(f"Processing uploaded image: {filename}")
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            logger.info(f"Image loaded successfully: {image.size}")
            
        except Exception as e:
            logger.error(f"Failed to decode image: {str(e)}")
            return jsonify({"error": "Invalid image data"}), 400
        
        start_time = time.time()
        
        # Process the uploaded image to 3D using the depth_map function
        result = process_image_to_3d(image, prompt=None, use_huggingface=False)
        
        total_time = time.time() - start_time
        logger.info(f"Image processing completed in {total_time:.2f} seconds")
        
        if not result["success"]:
            logger.error(f"Failed to process uploaded image: {result.get('error')}")
            return jsonify({
                "error": result.get('error', "Failed to process uploaded image"),
                "success": False,
                "processing_time": total_time
            }), 500
        
        # Return URLs for the 3D model and depth map
        return jsonify({
            "model_url": result["model_url"],           # URL of the 3D model file
            "depth_map_url": result["depth_map_url"],   # URL of the depth map
            "success": True,
            "processing_time": total_time
        })
        
    except Exception as e:
        logger.error(f"Upload endpoint error: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)
