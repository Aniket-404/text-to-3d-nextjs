import os
import json
import time
from PIL import Image, ImageDraw
import cloudinary
import cloudinary.uploader
import logging
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
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
    """Generate an image from a text prompt and create its 3D model"""
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        logger.info(f"Processing prompt: {prompt}")
        
        # Generate image and process to 3D using the unified function from depth_map.py
        result = process_image_to_3d(None, prompt=prompt, use_huggingface=True)
        
        if not result["success"]:
            logger.error(f"Failed to process image: {result.get('error')}")
            return jsonify({
                "error": result.get('error', "Failed to process image"),
                "success": False
            }), 500
        
        # Return URLs for both the generated image and 3D model
        return jsonify({
            "image_url": result["generated_image_url"],  # URL of the generated image
            "model_url": result["model_url"],           # URL of the 3D model file
            "depth_map_url": result["depth_map_url"],   # URL of the depth map (optional)
            "success": True
        })
            
    except Exception as e:
        logger.error(f"Failed to process request: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    """Upload an image and convert it to 3D model"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    if not file.filename.lower().endswith(tuple(allowed_extensions)):
        return jsonify({"error": "Invalid file type. Please upload an image file."}), 400
    
    try:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        temp_filename = f"upload_{timestamp}_{filename}"
        temp_path = os.path.join(TEMP_DIR, temp_filename)
        file.save(temp_path)
        
        logger.info(f"Uploaded file saved to: {temp_path}")
        
        # Upload to Cloudinary first
        upload_result = cloudinary.uploader.upload(
            temp_path,
            public_id=f"text-to-3d-web/uploads/{filename}_{timestamp}",
            resource_type="image",
            unique_filename=True,
            overwrite=True,
            quality="auto"
        )
        
        cloudinary_url = upload_result["secure_url"]
        logger.info(f"Uploaded to Cloudinary: {cloudinary_url}")
        
        # Clean up temporary file
        os.remove(temp_path)
        
        # Process the uploaded image to 3D (no prompt, so no image generation)
        result = process_image_to_3d(cloudinary_url, prompt=None, use_huggingface=False)
        
        if not result["success"]:
            logger.error(f"Failed to process uploaded image: {result.get('error')}")
            return jsonify({
                "error": result.get('error', "Failed to process uploaded image"),
                "success": False
            }), 500
        
        # Return URLs for both the uploaded image and 3D model
        return jsonify({
            "image_url": cloudinary_url,                    # URL of the uploaded image
            "model_url": result["model_url"],               # URL of the 3D model file
            "depth_map_url": result.get("depth_map_url"),   # URL of the depth map (optional)
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Failed to process uploaded image: {str(e)}", exc_info=True)
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            "error": str(e),
            "success": False
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



if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)
