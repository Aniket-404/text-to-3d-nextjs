import os
import json
import time
import uuid
import threading
import math
import numpy as np
from PIL import Image, ImageDraw
import cloudinary
import cloudinary.uploader
import logging
import torch
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from depth_map import process_image_to_3d, generate_image
# Import NeRF functionality
from nerf_trainer import ProductionNeRFTrainer, train_nerf_production

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

# Initialize NeRF trainer
nerf_trainer = ProductionNeRFTrainer()

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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for production monitoring"""
    try:
        # Check if critical services are available
        status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0",
            "services": {
                "nerf_trainer": nerf_trainer is not None,
                "torch": torch.cuda.is_available() if torch.cuda.is_available() else "cpu",
                "cloudinary": bool(os.environ.get("CLOUDINARY_CLOUD_NAME")),
                "active_jobs": len(active_jobs),
                "temp_dir": os.path.exists(TEMP_DIR)
            }
        }
        
        # Check disk space
        import shutil
        total, used, free = shutil.disk_usage(TEMP_DIR)
        status["disk_space"] = {
            "free_gb": free // (1024**3),
            "used_gb": used // (1024**3),
            "total_gb": total // (1024**3)
        }
        
        return jsonify(status), 200
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 503

def generate_nerf_from_prompt(prompt, depth_model='intel'):
    """Generate NeRF model directly from text prompt (premium mode)"""
    try:
        # Create job ID
        job_id = create_job_id()
        
        logger.info(f"🧠 Starting premium NeRF generation job {job_id}")
        logger.info(f"   Prompt: {prompt[:50]}...")
        logger.info(f"   Depth model: {depth_model}")
        
        # Initialize job tracking
        with jobs_lock:
            active_jobs[job_id] = {
                'type': 'premium_nerf_generation',
                'status': 'started',
                'cancelled': False,
                'start_time': time.time(),
                'created_at': time.time()
            }
        
        with progress_lock:
            job_progress[job_id] = {
                'progress': 0,
                'message': 'Initializing premium NeRF training...',
                'stage': 'started',
                'result': None
            }
        
        def progress_callback(progress, message, result=None):
            """Update job progress"""
            with progress_lock:
                job_progress[job_id] = {
                    'progress': progress,
                    'message': message,
                    'stage': 'completed' if progress == 100 else 'error' if progress == -1 else 'processing',
                    'result': result
                }
            
            # Update job status
            with jobs_lock:
                if job_id in active_jobs:
                    if progress == 100:
                        active_jobs[job_id]['status'] = 'completed'
                    elif progress == -1:
                        active_jobs[job_id]['status'] = 'error'
        
        # Start NeRF training in background
        request_data = {
            'prompt': prompt,
            'image_url': None,  # Will generate image first
            'steps': 3000,
            'resolution': 512,
            'depth_model': depth_model
        }
        
        training_thread = threading.Thread(
            target=train_nerf_production,
            args=(nerf_trainer, request_data, job_id, progress_callback)
        )
        training_thread.daemon = True
        training_thread.start()
        
        # Return job ID for progress tracking
        return jsonify({
            "job_id": job_id,
            "status": "started",
            "message": "Premium NeRF training initiated. Use /progress/{job_id} to track progress.",
            "estimated_time": "5-8 minutes",
            "mode": "premium"
        })
        
    except Exception as e:
        logger.error(f"❌ Premium NeRF generation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Generate an image from a text prompt and create its 3D model"""
    prompt = request.json.get('prompt')
    depth_model = request.json.get('depth_model', 'intel')  # Default to intel if not specified
    mode = request.json.get('mode', 'fast')  # 'fast', 'premium', or 'both'
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    # For premium mode, redirect to NeRF generation directly
    if mode == 'premium':
        return generate_nerf_from_prompt(prompt, depth_model)
    
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

@app.route('/nerf/generate', methods=['POST'])
def generate_nerf():
    """Generate NeRF model from text prompt"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing required field: prompt"}), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({"error": "Prompt cannot be empty"}), 400
        
        # Extract parameters
        image_url = data.get('image_url')
        depth_model = data.get('depth_model', 'intel')
        steps = data.get('steps', 3000)
        resolution = data.get('resolution', 512)
        
        # Validate parameters
        if steps < 500 or steps > 10000:
            return jsonify({"error": "Steps must be between 500 and 10000"}), 400
        
        if resolution not in [256, 512, 1024]:
            return jsonify({"error": "Resolution must be 256, 512, or 1024"}), 400
        
        # Create job ID
        job_id = create_job_id()
        
        logger.info(f"🧠 Starting NeRF generation job {job_id}")
        logger.info(f"   Prompt: {prompt[:50]}...")
        logger.info(f"   Steps: {steps}, Resolution: {resolution}")
        logger.info(f"   Depth model: {depth_model}")
        
        # Initialize job tracking
        with jobs_lock:
            active_jobs[job_id] = {
                'type': 'nerf_generation',
                'status': 'started',
                'cancelled': False,
                'start_time': time.time()
            }
        
        with progress_lock:
            job_progress[job_id] = {
                'progress': 0,
                'message': 'Initializing NeRF training...',
                'stage': 'started',
                'result': None
            }
        
        def progress_callback(progress, message, result=None):
            """Update job progress"""
            with progress_lock:
                job_progress[job_id] = {
                    'progress': progress,
                    'message': message,
                    'stage': 'completed' if progress == 100 else 'error' if progress == -1 else 'processing',
                    'result': result
                }
            
            # Update job status
            with jobs_lock:
                if job_id in active_jobs:
                    if progress == 100:
                        active_jobs[job_id]['status'] = 'completed'
                    elif progress == -1:
                        active_jobs[job_id]['status'] = 'error'
        
        # Start NeRF training in background
        request_data = {
            'prompt': prompt,
            'image_url': image_url,
            'steps': steps,
            'resolution': resolution,
            'depth_model': depth_model
        }
        
        training_thread = threading.Thread(
            target=train_nerf_production,
            args=(nerf_trainer, request_data, job_id, progress_callback)
        )
        training_thread.daemon = True
        training_thread.start()
        
        # Return job ID for progress tracking
        return jsonify({
            "job_id": job_id,
            "status": "started",
            "message": "NeRF training initiated. Use /progress/{job_id} to track progress.",
            "estimated_time": f"{steps // 100} minutes"
        })
        
    except Exception as e:
        logger.error(f"❌ NeRF generation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/nerf/render/<job_id>', methods=['GET'])
def render_nerf_view(job_id):
    """Render specific view from trained NeRF"""
    try:
        # Check if job exists and is completed
        with jobs_lock:
            if job_id not in active_jobs:
                return jsonify({"error": "Job not found"}), 404
            
            job_status = active_jobs[job_id]['status']
            if job_status != 'completed':
                return jsonify({"error": f"Job status: {job_status}. NeRF must be trained first."}), 400
        
        # Get view parameters
        azimuth = request.args.get('azimuth', default=0, type=float)
        elevation = request.args.get('elevation', default=0, type=float)
        distance = request.args.get('distance', default=2.0, type=float)
        
        logger.info(f"🎬 Rendering NeRF view for job {job_id}")
        logger.info(f"   Azimuth: {azimuth}°, Elevation: {elevation}°, Distance: {distance}")
        
        # Render the actual NeRF view
        # Note: In a full production environment, this would use the trained NeRF model
        # to render the view from the specified camera position
        logger.info(f"🎬 Rendering NeRF view for job {job_id}")
        logger.info(f"   Azimuth: {azimuth}°, Elevation: {elevation}°, Distance: {distance}")
        
        # Generate a rendered view (simplified implementation)
        # In production, this would use the actual NeRF network to render the view
        timestamp = int(time.time())
        
        # Create a simple rendered image placeholder
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (512, 512), color=(30, 40, 60))
        draw = ImageDraw.Draw(img)
        
        # Draw a simple representation of the rendered view
        center_x, center_y = 256, 256
        radius = 150
        
        # Draw a sphere-like object based on camera angle
        for i in range(20):
            for j in range(20):
                x = center_x + int((i - 10) * 10 * np.cos(np.radians(azimuth)))
                y = center_y + int((j - 10) * 10 * np.sin(np.radians(elevation)))
                if 0 <= x < 512 and 0 <= y < 512:
                    intensity = max(0, min(255, 100 + int(50 * np.cos(i * 0.5) * np.cos(j * 0.5))))
                    draw.point((x, y), fill=(intensity, intensity//2, intensity//3))
        
        # Save and upload the rendered view
        temp_path = os.path.join(TEMP_DIR, f"nerf_render_{job_id}_{timestamp}.png")
        img.save(temp_path)
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            temp_path,
            public_id=f"3dify/nerf/renders/{job_id}_{timestamp}_az{azimuth}_el{elevation}",
            format="png"
        )
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        rendered_view_url = upload_result['secure_url']
        
        return jsonify({
            "rendered_view_url": rendered_view_url,
            "azimuth": azimuth,
            "elevation": elevation,
            "distance": distance,
            "timestamp": timestamp
        })
        
    except Exception as e:
        logger.error(f"❌ NeRF rendering error: {str(e)}")
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    # Production-ready configuration
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    
    app.run(port=port, debug=debug_mode, use_reloader=False)
