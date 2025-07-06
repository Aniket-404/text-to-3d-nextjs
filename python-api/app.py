import os
import json
import time
import uuid
import threading
from PIL import Image, ImageDraw
import cloudinary
import cloudinary.uploader
import logging
import torch
from flask import Flask, request, jsonify, Response
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

# Global dictionary to track active jobs and their cancellation status
active_jobs = {}
jobs_lock = threading.Lock()

# Global dictionary to track job progress
job_progress = {}
progress_lock = threading.Lock()

def create_job_id():
    """Generate a unique job ID"""
    return str(uuid.uuid4())

def is_job_cancelled(job_id):
    """Check if a job has been cancelled"""
    with jobs_lock:
        return active_jobs.get(job_id, {}).get('cancelled', False)

def cancel_job(job_id):
    """Cancel a job"""
    with jobs_lock:
        if job_id in active_jobs:
            active_jobs[job_id]['cancelled'] = True
            logger.info(f"Job {job_id} has been cancelled")
            return True
        return False

def register_job(job_id):
    """Register a new job"""
    with jobs_lock:
        active_jobs[job_id] = {'cancelled': False, 'created_at': time.time()}
    with progress_lock:
        job_progress[job_id] = {'stage': 'starting', 'progress': 0, 'message': 'Initializing...'}

def update_job_progress(job_id, stage, progress, message):
    """Update job progress"""
    with progress_lock:
        if job_id in job_progress:
            job_progress[job_id] = {
                'stage': stage,
                'progress': progress,
                'message': message,
                'timestamp': time.time()
            }
            logger.info(f"Job {job_id} progress: {stage} - {progress}% - {message}")

def get_job_progress(job_id):
    """Get current job progress"""
    with progress_lock:
        return job_progress.get(job_id, {'stage': 'unknown', 'progress': 0, 'message': 'Not found'})

def cleanup_job(job_id):
    """Clean up a completed job"""
    with jobs_lock:
        if job_id in active_jobs:
            del active_jobs[job_id]
    with progress_lock:
        if job_id in job_progress:
            del job_progress[job_id]

def cleanup_old_jobs():
    """Clean up jobs older than 30 minutes"""
    current_time = time.time()
    with jobs_lock:
        jobs_to_remove = []
        for job_id, job_data in active_jobs.items():
            if current_time - job_data['created_at'] > 1800:  # 30 minutes
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del active_jobs[job_id]
            logger.info(f"Cleaned up old job: {job_id}")
    
    with progress_lock:
        for job_id in jobs_to_remove:
            if job_id in job_progress:
                del job_progress[job_id]

# Run cleanup every 5 minutes
def periodic_cleanup():
    import threading
    def cleanup_worker():
        while True:
            time.sleep(300)  # 5 minutes
            cleanup_old_jobs()
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()

# Start periodic cleanup
periodic_cleanup()

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
    depth_model = request.json.get('depth_model', 'intel')  # Default to intel if not specified
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    # Create a unique job ID
    job_id = create_job_id()
    register_job(job_id)
    
    try:
        logger.info(f"Processing prompt: {prompt} (Job ID: {job_id})")
        update_job_progress(job_id, 'starting', 5, 'Starting image generation...')
        
        # Generate image and process to 3D using the unified function from depth_map.py
        result = process_image_to_3d(None, prompt=prompt, use_huggingface=True, job_id=job_id, depth_model=depth_model)
        
        # Check if job was cancelled during processing
        if is_job_cancelled(job_id):
            logger.info(f"Job {job_id} was cancelled during processing")
            return jsonify({
                "error": "Request was cancelled",
                "success": False,
                "cancelled": True
            }), 499  # Client Closed Request
        
        if not result["success"]:
            logger.error(f"Failed to process image: {result.get('error')}")
            update_job_progress(job_id, 'error', 0, f"Error: {result.get('error')}")
            return jsonify({
                "error": result.get('error', "Failed to process image"),
                "success": False
            }), 500
        
        update_job_progress(job_id, 'completed', 100, 'Generation completed successfully!')
        
        # Return URLs for both the generated image and 3D model
        return jsonify({
            "image_url": result["generated_image_url"],  # URL of the generated image
            "model_url": result["model_url"],           # URL of the 3D model file
            "depth_map_url": result["depth_map_url"],   # URL of the depth map (optional)
            "success": True,
            "job_id": job_id
        })
            
    except Exception as e:
        logger.error(f"Failed to process request: {str(e)}", exc_info=True)
        update_job_progress(job_id, 'error', 0, f"Error: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500
    finally:
        cleanup_job(job_id)

@app.route('/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    """Get the progress of a job"""
    try:
        progress = get_job_progress(job_id)
        return jsonify(progress)
    except Exception as e:
        return jsonify({"error": f"Failed to get progress: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    """Upload an image and convert it to 3D model"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    depth_model = request.form.get('depth_model', 'intel')  # Default to intel if not specified
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    if not file.filename.lower().endswith(tuple(allowed_extensions)):
        return jsonify({"error": "Invalid file type. Please upload an image file."}), 400
    
    # Create a unique job ID
    job_id = create_job_id()
    register_job(job_id)
    
    try:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        temp_filename = f"upload_{timestamp}_{filename}"
        temp_path = os.path.join(TEMP_DIR, temp_filename)
        file.save(temp_path)
        
        logger.info(f"Uploaded file saved to: {temp_path} (Job ID: {job_id})")
        update_job_progress(job_id, 'uploading', 10, 'Uploading image to cloud...')
        
        # Check if job was cancelled early
        if is_job_cancelled(job_id):
            os.remove(temp_path)
            logger.info(f"Job {job_id} was cancelled before processing")
            return jsonify({
                "error": "Request was cancelled",
                "success": False,
                "cancelled": True
            }), 499
        
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
        update_job_progress(job_id, 'uploaded', 20, 'Image uploaded, starting 3D conversion...')
        
        # Clean up temporary file
        os.remove(temp_path)
        
        # Check if job was cancelled after upload
        if is_job_cancelled(job_id):
            logger.info(f"Job {job_id} was cancelled after Cloudinary upload")
            return jsonify({
                "error": "Request was cancelled",
                "success": False,
                "cancelled": True
            }), 499
        
        # Process the uploaded image to 3D (no prompt, so no image generation)
        result = process_image_to_3d(cloudinary_url, prompt=None, use_huggingface=False, job_id=job_id, depth_model=depth_model)
        
        # Check if job was cancelled during processing
        if is_job_cancelled(job_id):
            logger.info(f"Job {job_id} was cancelled during processing")
            return jsonify({
                "error": "Request was cancelled",
                "success": False,
                "cancelled": True
            }), 499
        
        if not result["success"]:
            logger.error(f"Failed to process uploaded image: {result.get('error')}")
            update_job_progress(job_id, 'error', 0, f"Error: {result.get('error')}")
            return jsonify({
                "error": result.get('error', "Failed to process uploaded image"),
                "success": False
            }), 500
        
        update_job_progress(job_id, 'completed', 100, 'Upload and conversion completed successfully!')
        
        # Return URLs for both the uploaded image and 3D model
        return jsonify({
            "image_url": cloudinary_url,                    # URL of the uploaded image
            "model_url": result["model_url"],               # URL of the 3D model file
            "depth_map_url": result.get("depth_map_url"),   # URL of the depth map (optional)
            "success": True,
            "job_id": job_id
        })
        
    except Exception as e:
        logger.error(f"Failed to process uploaded image: {str(e)}", exc_info=True)
        update_job_progress(job_id, 'error', 0, f"Error: {str(e)}")
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            "error": str(e),
            "success": False
        }), 500
    finally:
        cleanup_job(job_id)

@app.route('/cancel', methods=['POST'])
def cancel_request():
    """Cancel a running job"""
    data = request.json
    job_id = data.get('job_id')
    
    if not job_id:
        return jsonify({"error": "No job_id provided"}), 400
    
    try:
        if cancel_job(job_id):
            return jsonify({"message": f"Job {job_id} cancelled successfully", "success": True})
        else:
            return jsonify({"error": f"Job {job_id} not found or already completed", "success": False}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to cancel job: {str(e)}", "success": False}), 500

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
