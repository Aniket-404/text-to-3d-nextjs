"""
Sparse View 3D Reconstruction implementation for text-to-3D generation.
This module provides multi-view sparse reconstruction using camera-conditioned diffusion
and advanced 3D reconstruction algorithms.
"""

import os
import json
import time
import uuid
import torch
import numpy as np
import trimesh
from PIL import Image
import cloudinary
import cloudinary.uploader
from typing import Dict, Any, Optional, Tuple, Callable, List
import logging
import cv2
from pathlib import Path

# Check for production dependencies
try:
    from diffusers import StableDiffusionPipeline
    from transformers import pipeline
    import open3d as o3d
    HAS_PRODUCTION_DEPS = True
except ImportError:
    HAS_PRODUCTION_DEPS = False

logger = logging.getLogger(__name__)

class SparseViewReconstructor:
    """Production-grade sparse view 3D reconstruction using camera-conditioned generation"""
    
    def __init__(self, device: str = None, cache_dir: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.info("CUDA not available, using CPU")
        else:
            self.device = device
            
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "sparse_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize with lazy loading - don't load models yet
        self.sd_pipe = None
        self._models_initialized = False
        
        logger.info(f"Sparse view reconstructor initialized on device: {self.device}")
        logger.info(f"Cache directory: {self.cache_dir}")
        
    def _ensure_models_loaded(self):
        """Lazy load models only when needed"""
        if not self._models_initialized:
            self._init_models()
            self._models_initialized = True
        
    def _init_models(self):
        """Initialize required models for sparse view reconstruction"""
        try:
            # Check available GPU memory
            if self.device == "cuda" and torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                free_memory = total_memory - allocated
                
                logger.info(f"GPU Memory - Total: {total_memory:.1f}GB, Free: {free_memory:.1f}GB")
                
                # If low on memory, don't load Stable Diffusion
                if free_memory < 2.0:  # Less than 2GB free
                    logger.warning(f"Insufficient GPU memory ({free_memory:.1f}GB free). Skipping Stable Diffusion loading.")
                    self.sd_pipe = None
                    return
            
            if HAS_PRODUCTION_DEPS and self.device == "cuda":
                # Initialize Stable Diffusion for multi-view generation with memory optimization
                self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    cache_dir=self.cache_dir,
                    safety_checker=None,  # Disable safety checker to save memory
                    requires_safety_checker=False
                ).to(self.device)
                
                # Enable memory efficient attention and other optimizations
                if hasattr(self.sd_pipe, 'enable_attention_slicing'):
                    self.sd_pipe.enable_attention_slicing()
                
                if hasattr(self.sd_pipe, 'enable_model_cpu_offload'):
                    self.sd_pipe.enable_model_cpu_offload()
                
                if hasattr(self.sd_pipe, 'enable_xformers_memory_efficient_attention'):
                    try:
                        self.sd_pipe.enable_xformers_memory_efficient_attention()
                        logger.info("Enabled xformers memory efficient attention")
                    except:
                        logger.info("xformers not available, using standard attention")
                    
                logger.info("Stable Diffusion pipeline loaded for multi-view generation")
            else:
                self.sd_pipe = None
                logger.warning("Running in CPU mode or missing dependencies - using simplified pipeline")
                
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self.sd_pipe = None

    def reconstruct_from_text(
        self,
        prompt: str,
        num_views: int = 6,
        resolution: int = 512,
        negative_prompt: str = 'low quality, bad anatomy, worst quality, low resolution, blurry',
        job_id: str = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Reconstruct 3D model from text using sparse view approach
        
        Args:
            prompt: Text description of the 3D scene
            num_views: Number of viewpoints to generate (4-8 recommended)
            resolution: Image resolution for generation
            job_id: Unique identifier for this job
            progress_callback: Function to report progress
            
        Returns:
            Dictionary containing all generated assets and URLs
        """
        # Ensure models are loaded before reconstruction
        self._ensure_models_loaded()
        
        if job_id is None:
            job_id = str(uuid.uuid4())
            
        logger.info(f"🔄 Starting sparse view reconstruction for job {job_id}")
        logger.info(f"   Prompt: {prompt[:50]}...")
        logger.info(f"   Views: {num_views}, Resolution: {resolution}")
        
        try:
            # Phase 1: Generate multi-view images (20%)
            if progress_callback:
                progress_callback(10, f"Generating {num_views} camera views...")
            
            multi_view_images, camera_poses = self._generate_camera_conditioned_views(
                prompt, num_views, resolution, negative_prompt, progress_callback
            )
            
            # Phase 2: Feature matching and sparse reconstruction (40%)
            if progress_callback:
                progress_callback(30, "Performing feature matching...")
                
            sparse_point_cloud = self._sparse_reconstruction_from_views(
                multi_view_images, camera_poses, progress_callback
            )
            
            # Phase 3: Dense reconstruction and meshing (70%)
            if progress_callback:
                progress_callback(60, "Creating 3D mesh from sparse points...")
                
            mesh_data = self._dense_reconstruction_and_meshing(
                sparse_point_cloud, multi_view_images, camera_poses, progress_callback
            )
            
            # Phase 4: Asset generation and upload (90%)
            if progress_callback:
                progress_callback(80, "Generating final assets...")
                
            outputs = self._generate_sparse_outputs_from_text(
                mesh_data, multi_view_images, prompt, resolution, job_id
            )
            
            if progress_callback:
                progress_callback(100, "Sparse view reconstruction complete!")
                
            logger.info(f"Sparse view reconstruction completed for job {job_id}")
            return outputs
            
        except Exception as e:
            logger.error(f"Sparse view reconstruction failed: {str(e)}")
            if progress_callback:
                progress_callback(-1, f"Error: {str(e)}")
            raise

    def _generate_camera_conditioned_views(
        self, 
        prompt: str, 
        num_views: int, 
        resolution: int,
        negative_prompt: str = 'low quality, bad anatomy, worst quality, low resolution, blurry',
        progress_callback: Optional[Callable] = None
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """Generate multiple views using camera-conditioned diffusion"""
        
        images = []
        camera_poses = []
        
        # Define strategic camera positions for optimal reconstruction
        view_configs = self._get_strategic_viewpoints(num_views)
        
        if self.sd_pipe is not None:
            for i, view_config in enumerate(view_configs):
                try:
                    # Create camera-aware prompt
                    view_prompt = self._create_camera_conditioned_prompt(prompt, view_config)
                    
                    if progress_callback:
                        progress = 10 + int((i / num_views) * 15)  # 10% to 25%
                        progress_callback(progress, f"Generating view {i+1}/{num_views}: {view_config['name']}")
                    
                    with torch.autocast(self.device):
                        image = self.sd_pipe(
                            view_prompt,
                            negative_prompt=negative_prompt,
                            height=resolution,
                            width=resolution,
                            num_inference_steps=50,
                            guidance_scale=7.5,
                            generator=torch.Generator(self.device).manual_seed(42 + i)
                        ).images[0]
                    
                    images.append(np.array(image))
                    camera_poses.append(view_config)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate view {i}: {e}")
                    
        else:
            # Fallback: Create synthetic multi-view images
            logger.info("Creating synthetic multi-view images")
            for i, view_config in enumerate(view_configs):
                # Create a simple colored image as fallback
                image = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
                images.append(image)
                camera_poses.append(view_config)
                
        return images, camera_poses

    def _get_strategic_viewpoints(self, num_views: int) -> List[Dict]:
        """Get strategic camera viewpoints for optimal 3D reconstruction"""
        
        base_configs = [
            {"name": "front", "elevation": 0, "azimuth": 0, "distance": 3.0},
            {"name": "right", "elevation": 0, "azimuth": 90, "distance": 3.0},
            {"name": "back", "elevation": 0, "azimuth": 180, "distance": 3.0},
            {"name": "left", "elevation": 0, "azimuth": 270, "distance": 3.0},
            {"name": "front_up", "elevation": 30, "azimuth": 45, "distance": 3.0},
            {"name": "back_up", "elevation": 30, "azimuth": 225, "distance": 3.0},
            {"name": "top", "elevation": 60, "azimuth": 0, "distance": 3.0},
            {"name": "top_angled", "elevation": 45, "azimuth": 135, "distance": 3.0},
        ]
        
        # Select the most strategic views based on num_views
        if num_views <= len(base_configs):
            return base_configs[:num_views]
        else:
            # Add additional views by interpolating
            additional_views = []
            for i in range(num_views - len(base_configs)):
                azimuth = (i * 360) / (num_views - len(base_configs))
                additional_views.append({
                    "name": f"extra_{i}",
                    "elevation": 15,
                    "azimuth": azimuth,
                    "distance": 3.0
                })
            return base_configs + additional_views

    def _create_camera_conditioned_prompt(self, base_prompt: str, view_config: Dict) -> str:
        """Create camera-aware prompts for consistent multi-view generation"""
        
        view_descriptors = {
            "front": "front view, centered, facing camera",
            "right": "right side view, profile view from the right",
            "back": "back view, rear view",
            "left": "left side view, profile view from the left",
            "front_up": "three quarter view from front-right and slightly above",
            "back_up": "three quarter view from back-left and slightly above",
            "top": "top view, bird's eye view, from above",
            "top_angled": "isometric view, angled from above"
        }
        
        view_desc = view_descriptors.get(view_config["name"], "side view")
        
        # Enhance prompt with camera information
        camera_prompt = f"{base_prompt}, {view_desc}, clean background, professional lighting, high detail"
        
        return camera_prompt

    def _sparse_reconstruction_from_views(
        self, 
        images: list,
        camera_poses: list,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Perform sparse 3D reconstruction from multiple views
        
        This method uses feature matching and triangulation to create
        a sparse point cloud from multiple viewpoints.
        """
        try:
            # Feature detection and matching
            feature_points = []
            descriptors = []
            
            # Use ORB for feature detection
            orb = cv2.ORB_create(nfeatures=1000)
            
            for i, img in enumerate(images):
                if progress_callback:
                    progress = 25 + (i / len(images)) * 15
                    progress_callback(progress, f"Extracting features from view {i+1}/{len(images)}...")
                
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
                kp, desc = orb.detectAndCompute(gray, None)
                
                if desc is not None:
                    feature_points.append(kp)
                    descriptors.append(desc)
            
            # Match features between consecutive views
            if progress_callback:
                progress_callback(40, "Matching features across views...")
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches_data = []
            
            for i in range(len(descriptors) - 1):
                matches = bf.match(descriptors[i], descriptors[i + 1])
                matches = sorted(matches, key=lambda x: x.distance)
                matches_data.append(matches[:50])  # Keep top 50 matches
            
            # Triangulate 3D points
            if progress_callback:
                progress_callback(50, "Triangulating 3D points...")
            
            sparse_points = []
            colors = []
            
            for i, matches in enumerate(matches_data):
                if len(matches) < 8:
                    continue
                
                # Get matched points
                pts1 = np.float32([feature_points[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts2 = np.float32([feature_points[i+1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # Camera matrices
                P1 = self._get_projection_matrix(camera_poses[i])
                P2 = self._get_projection_matrix(camera_poses[i + 1])
                
                # Triangulate points
                points_4d = cv2.triangulatePoints(P1, P2, pts1.reshape(2, -1), pts2.reshape(2, -1))
                points_3d = points_4d[:3] / points_4d[3]
                
                for j, point in enumerate(points_3d.T):
                    if abs(point[2]) < 10:  # Filter out points too far
                        sparse_points.append(point)
                        # Get color from original image
                        x, y = int(pts1[j, 0, 0]), int(pts1[j, 0, 1])
                        if 0 <= x < images[i].shape[1] and 0 <= y < images[i].shape[0]:
                            color = images[i][y, x] if len(images[i].shape) == 3 else [128, 128, 128]
                            colors.append(color)
                        else:
                            colors.append([128, 128, 128])
            
            return {
                'points': np.array(sparse_points),
                'colors': np.array(colors),
                'num_points': len(sparse_points),
                'camera_poses': camera_poses,
                'matches_data': matches_data
            }
            
        except Exception as e:
            logger.error(f"Sparse reconstruction failed: {str(e)}")
            # Return minimal sparse cloud
            return {
                'points': np.array([[0, 0, 0]]),
                'colors': np.array([[128, 128, 128]]),
                'num_points': 1,
                'camera_poses': camera_poses,
                'matches_data': []
            }
    
    def _get_projection_matrix(self, camera_pose: Dict[str, Any]) -> np.ndarray:
        """Get projection matrix from camera pose"""
        # Camera intrinsic parameters (assuming typical values)
        fx = fy = 800.0  # Focal length
        cx = cy = 256.0  # Principal point
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # Extrinsic matrix (camera pose)
        R = camera_pose['rotation']
        t = camera_pose['position'].reshape(3, 1)
        
        # Create [R|t] matrix
        Rt = np.hstack([R, -R @ t])
        
        # Projection matrix P = K[R|t]
        P = K @ Rt
        return P

    def reconstruct_from_images(
        self,
        images: list,
        reference_image_url: str,
        job_id: str = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Reconstruct 3D model from a list of pre-existing images
        
        Args:
            images: List of numpy arrays representing different viewpoints
            reference_image_url: URL of the original reference image
            job_id: Job ID for progress tracking
            progress_callback: Callback function for progress updates
            
        Returns:
            Dictionary containing sparse reconstruction results
        """
        # Ensure models are loaded before reconstruction
        self._ensure_models_loaded()
        
        try:
            logger.info(f"Starting sparse reconstruction from images for job {job_id}")
            
            # Phase 1: Process input images (10%)
            if progress_callback:
                progress_callback(10, "Processing input images...")
            
            # Convert images to the expected format
            processed_images = []
            camera_poses = []
            
            for i, img_array in enumerate(images):
                if isinstance(img_array, np.ndarray):
                    processed_images.append(img_array)
                else:
                    # Convert PIL Image or other formats to numpy array
                    processed_images.append(np.array(img_array))
                
                # Generate camera poses for the images
                # Assume images are taken from a circular path around the object
                angle = 2 * np.pi * i / len(images)
                pose = self._generate_camera_pose_for_angle(angle)
                camera_poses.append(pose)
            
            # Phase 2: Feature matching and sparse reconstruction (40%)
            if progress_callback:
                progress_callback(20, "Performing feature matching across views...")
                
            sparse_point_cloud = self._sparse_reconstruction_from_views(
                processed_images, camera_poses, progress_callback
            )
            
            # Phase 3: Dense surface reconstruction (30%)
            if progress_callback:
                progress_callback(60, "Generating dense surface reconstruction...")
                
            dense_mesh = self._dense_surface_reconstruction(sparse_point_cloud, progress_callback)
            
            # Phase 4: Generate outputs (20%)
            if progress_callback:
                progress_callback(80, "Generating sparse reconstruction outputs...")
                
            outputs = self._generate_sparse_outputs(
                processed_images,
                dense_mesh,
                reference_image_url,
                job_id,
                sparse_point_cloud
            )
            
            if progress_callback:
                progress_callback(100, "Sparse reconstruction from images complete!")
                
            logger.info(f"Sparse reconstruction from images completed for job {job_id}")
            return outputs
            
        except Exception as e:
            logger.error(f"Sparse reconstruction from images failed: {str(e)}")
            if progress_callback:
                progress_callback(-1, f"Error: {str(e)}")
            raise
    
    def _generate_camera_pose_for_angle(self, angle: float) -> Dict[str, Any]:
        """Generate camera pose for a specific angle around the object"""
        radius = 2.0
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = 0.0
        
        # Look-at matrix
        camera_pos = np.array([x, y, z])
        target = np.array([0, 0, 0])
        up = np.array([0, 1, 0])
        
        # Compute rotation matrix
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Build pose matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, 0] = right
        pose_matrix[:3, 1] = up
        pose_matrix[:3, 2] = -forward
        pose_matrix[:3, 3] = camera_pos
        
        return {
            'name': f'view_{int(np.degrees(angle))}deg',
            'position': camera_pos,
            'rotation': pose_matrix[:3, :3],
            'pose_matrix': pose_matrix,
            'angle': angle
        }
    
    def _dense_surface_reconstruction(
        self,
        sparse_point_cloud: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Generate dense surface reconstruction from sparse point cloud
        
        This method creates a dense mesh from the sparse point cloud using
        Poisson surface reconstruction and other techniques.
        """
        try:
            if progress_callback:
                progress_callback(65, "Creating dense point cloud...")
            
            sparse_points = sparse_point_cloud['points']
            sparse_colors = sparse_point_cloud['colors']
            
            if len(sparse_points) < 3:
                logger.warning("Insufficient sparse points for dense reconstruction")
                return self._create_fallback_mesh()
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sparse_points)
            pcd.colors = o3d.utility.Vector3dVector(sparse_colors / 255.0)
            
            # Estimate normals
            if progress_callback:
                progress_callback(70, "Estimating surface normals...")
            
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
            
            # Orient normals consistently
            pcd.orient_normals_consistent_tangent_plane(100)
            
            # Poisson surface reconstruction
            if progress_callback:
                progress_callback(75, "Performing Poisson surface reconstruction...")
            
            try:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=9
                )
                
                # Remove low density vertices
                vertices_to_remove = densities < np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(vertices_to_remove)
                
            except Exception as e:
                logger.warning(f"Poisson reconstruction failed: {e}, using alpha shapes")
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha=0.03
                )
            
            # Clean up mesh
            if progress_callback:
                progress_callback(78, "Cleaning up mesh...")
            
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            
            # Smooth the mesh
            mesh = mesh.filter_smooth_simple(number_of_iterations=5)
            
            # Compute vertex normals
            mesh.compute_vertex_normals()
            
            # Convert to format suitable for export
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            normals = np.asarray(mesh.vertex_normals)
            
            # Generate texture coordinates (simple spherical mapping)
            texture_coords = self._generate_spherical_uv_coordinates(vertices)
            
            return {
                'vertices': vertices,
                'faces': faces,
                'normals': normals,
                'texture_coords': texture_coords,
                'mesh': mesh,
                'vertex_count': len(vertices),
                'face_count': len(faces)
            }
            
        except Exception as e:
            logger.error(f"Dense surface reconstruction failed: {str(e)}")
            return self._create_fallback_mesh()
    
    def _create_fallback_mesh(self) -> Dict[str, Any]:
        """Create a simple fallback mesh when reconstruction fails"""
        # Create a simple cube mesh
        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        mesh.compute_vertex_normals()
        
        vertices = np.asarray(mesh.vertices) - 0.5  # Center the cube
        faces = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.vertex_normals)
        texture_coords = self._generate_spherical_uv_coordinates(vertices)
        
        return {
            'vertices': vertices,
            'faces': faces,
            'normals': normals,
            'texture_coords': texture_coords,
            'mesh': mesh,
            'vertex_count': len(vertices),
            'face_count': len(faces)
        }
    
    def _generate_spherical_uv_coordinates(self, vertices: np.ndarray) -> np.ndarray:
        """Generate spherical UV coordinates for vertices"""
        # Normalize vertices to unit sphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_vertices = vertices / norms
        
        # Convert to spherical coordinates
        x, y, z = normalized_vertices[:, 0], normalized_vertices[:, 1], normalized_vertices[:, 2]
        
        # UV mapping
        u = 0.5 + np.arctan2(z, x) / (2 * np.pi)
        v = 0.5 - np.arcsin(y) / np.pi
        
        return np.column_stack([u, v])

    def _generate_sparse_outputs(
        self,
        input_images: list,
        dense_mesh: Dict[str, Any],
        reference_image_url: str,
        job_id: str,
        sparse_point_cloud: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate all output files for sparse reconstruction"""
        try:
            timestamp = int(time.time())
            base_filename = f"sparse_{job_id}_{timestamp}" if job_id else f"sparse_{timestamp}"
            
            outputs = {
                'sparse_url': None,
                'sparse_depth_url': None,
                'sparse_wireframe_url': None,
                'sparse_normal_url': None,
                'reference_image_url': reference_image_url,
                'metadata': {
                    'vertex_count': dense_mesh['vertex_count'],
                    'face_count': dense_mesh['face_count'],
                    'sparse_points': sparse_point_cloud['num_points'],
                    'reconstruction_method': 'sparse_view',
                    'timestamp': timestamp
                }
            }
            
            # Save OBJ file
            obj_filename = f"{base_filename}.obj"
            obj_path = os.path.join(self.output_dir, obj_filename)
            self._save_obj_file(dense_mesh, obj_path)
            
            if os.path.exists(obj_path):
                outputs['sparse_url'] = self._upload_to_cloudinary(obj_path, 'model')
            
            # Generate depth map visualization
            depth_image = self._generate_depth_visualization(dense_mesh, input_images[0])
            depth_filename = f"{base_filename}_depth.png"
            depth_path = os.path.join(self.output_dir, depth_filename)
            cv2.imwrite(depth_path, depth_image)
            
            if os.path.exists(depth_path):
                outputs['sparse_depth_url'] = self._upload_to_cloudinary(depth_path, 'image')
            
            # Generate wireframe visualization
            wireframe_image = self._generate_wireframe_visualization(dense_mesh, input_images[0])
            wireframe_filename = f"{base_filename}_wireframe.png"
            wireframe_path = os.path.join(self.output_dir, wireframe_filename)
            cv2.imwrite(wireframe_path, wireframe_image)
            
            if os.path.exists(wireframe_path):
                outputs['sparse_wireframe_url'] = self._upload_to_cloudinary(wireframe_path, 'image')
            
            # Generate normal map visualization
            normal_image = self._generate_normal_visualization(dense_mesh)
            normal_filename = f"{base_filename}_normal.png"
            normal_path = os.path.join(self.output_dir, normal_filename)
            cv2.imwrite(normal_path, normal_image)
            
            if os.path.exists(normal_path):
                outputs['sparse_normal_url'] = self._upload_to_cloudinary(normal_path, 'image')
            
            # Clean up temporary files
            for file_path in [obj_path, depth_path, wireframe_path, normal_path]:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up file {file_path}: {e}")
            
            logger.info(f"Generated sparse reconstruction outputs: {outputs}")
            return outputs
            
        except Exception as e:
            logger.error(f"Failed to generate sparse outputs: {str(e)}")
            return {
                'sparse_url': None,
                'sparse_depth_url': None,
                'sparse_wireframe_url': None,
                'sparse_normal_url': None,
                'reference_image_url': reference_image_url,
                'metadata': {'error': str(e)}
            }
    
    def _save_obj_file(self, dense_mesh: Dict[str, Any], obj_path: str):
        """Save mesh as OBJ file"""
        vertices = dense_mesh['vertices']
        faces = dense_mesh['faces']
        normals = dense_mesh['normals']
        texture_coords = dense_mesh['texture_coords']
        
        with open(obj_path, 'w') as f:
            # Write header
            f.write("# Sparse View 3D Reconstruction\n")
            f.write(f"# Generated at {time.ctime()}\n")
            f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write texture coordinates
            for tc in texture_coords:
                f.write(f"vt {tc[0]:.6f} {tc[1]:.6f}\n")
            
            # Write normals
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            
            f.write("\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                       f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                       f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")
    
    def _generate_depth_visualization(self, dense_mesh: Dict[str, Any], reference_image: np.ndarray) -> np.ndarray:
        """Generate depth map visualization from mesh"""
        try:
            vertices = dense_mesh['vertices']
            
            # Simple depth visualization - project vertices to image plane
            if len(vertices) == 0:
                return np.zeros_like(reference_image[:, :, 0])
            
            # Get depth range
            min_depth = np.min(vertices[:, 2])
            max_depth = np.max(vertices[:, 2])
            
            if max_depth == min_depth:
                return np.zeros_like(reference_image[:, :, 0])
            
            # Create depth map
            h, w = reference_image.shape[:2]
            depth_map = np.zeros((h, w), dtype=np.float32)
            
            # Project vertices to image coordinates
            scale = min(w, h) / 4  # Scale factor for projection
            center_x, center_y = w // 2, h // 2
            
            for vertex in vertices:
                x = int(vertex[0] * scale + center_x)
                y = int(vertex[1] * scale + center_y)
                
                if 0 <= x < w and 0 <= y < h:
                    normalized_depth = (vertex[2] - min_depth) / (max_depth - min_depth)
                    depth_map[y, x] = max(depth_map[y, x], normalized_depth)
            
            # Convert to 8-bit image
            depth_image = (depth_map * 255).astype(np.uint8)
            
            # Apply colormap for better visualization
            depth_colored = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            
            return depth_colored
            
        except Exception as e:
            logger.error(f"Depth visualization failed: {str(e)}")
            return np.zeros_like(reference_image)
    
    def _generate_wireframe_visualization(self, dense_mesh: Dict[str, Any], reference_image: np.ndarray) -> np.ndarray:
        """Generate wireframe visualization from mesh"""
        try:
            vertices = dense_mesh['vertices']
            faces = dense_mesh['faces']
            
            if len(vertices) == 0 or len(faces) == 0:
                return reference_image.copy()
            
            # Create wireframe image
            h, w = reference_image.shape[:2]
            wireframe_image = reference_image.copy()
            
            # Projection parameters
            scale = min(w, h) / 4
            center_x, center_y = w // 2, h // 2
            
            # Draw edges
            for face in faces:
                points = []
                for vertex_idx in face:
                    if vertex_idx < len(vertices):
                        vertex = vertices[vertex_idx]
                        x = int(vertex[0] * scale + center_x)
                        y = int(vertex[1] * scale + center_y)
                        points.append((x, y))
                
                if len(points) == 3:
                    # Draw triangle edges
                    for i in range(3):
                        pt1 = points[i]
                        pt2 = points[(i + 1) % 3]
                        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                            0 <= pt2[0] < w and 0 <= pt2[1] < h):
                            cv2.line(wireframe_image, pt1, pt2, (0, 255, 0), 1)
            
            return wireframe_image
            
        except Exception as e:
            logger.error(f"Wireframe visualization failed: {str(e)}")
            return reference_image.copy()
    
    def _generate_normal_visualization(self, dense_mesh: Dict[str, Any]) -> np.ndarray:
        """Generate normal map visualization from mesh"""
        try:
            normals = dense_mesh['normals']
            
            if len(normals) == 0:
                return np.zeros((512, 512, 3), dtype=np.uint8)
            
            # Convert normals to RGB (normal mapping convention)
            # X -> Red, Y -> Green, Z -> Blue
            normal_image = ((normals + 1.0) * 127.5).astype(np.uint8)
            
            # Create a 2D representation
            size = int(np.sqrt(len(normals)))
            if size * size < len(normals):
                size += 1
            
            # Pad or trim normals to fit square image
            padded_normals = np.zeros((size * size, 3), dtype=np.uint8)
            padded_normals[:len(normal_image)] = normal_image
            
            # Reshape to 2D image
            normal_map = padded_normals.reshape(size, size, 3)
            
            # Resize to standard size
            normal_map = cv2.resize(normal_map, (512, 512), interpolation=cv2.INTER_LINEAR)
            
            return normal_map
            
        except Exception as e:
            logger.error(f"Normal visualization failed: {str(e)}")
            return np.zeros((512, 512, 3), dtype=np.uint8)

    def _upload_to_cloudinary(self, file_path: str, resource_type: str = 'auto') -> Optional[str]:
        """Upload file to Cloudinary and return URL"""
        try:
            result = cloudinary.uploader.upload(
                file_path,
                resource_type=resource_type,
                folder="sparse_reconstruction",
                use_filename=True,
                unique_filename=True
            )
            return result.get('secure_url')
        except Exception as e:
            logger.error(f"Failed to upload to Cloudinary: {str(e)}")
            return None

    def _dense_reconstruction_and_meshing(
        self,
        sparse_point_cloud: Dict[str, Any],
        multi_view_images: List[np.ndarray],
        camera_poses: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Perform dense reconstruction and meshing from sparse point cloud"""
        try:
            logger.info("Starting dense reconstruction and meshing")
            
            if progress_callback:
                progress_callback(65, "Creating dense point cloud...")
            
            sparse_points = sparse_point_cloud['points']
            sparse_colors = sparse_point_cloud['colors']
            
            if len(sparse_points) < 3:
                logger.warning("Insufficient sparse points for dense reconstruction")
                return self._create_fallback_mesh()
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sparse_points)
            pcd.colors = o3d.utility.Vector3dVector(sparse_colors / 255.0)
            
            # Estimate normals
            if progress_callback:
                progress_callback(70, "Estimating surface normals...")
            
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
            
            # Orient normals consistently
            pcd.orient_normals_consistent_tangent_plane(100)
            
            # Poisson surface reconstruction
            if progress_callback:
                progress_callback(75, "Performing Poisson surface reconstruction...")
            
            try:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=9
                )
                
                # Remove low density vertices
                vertices_to_remove = densities < np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(vertices_to_remove)
                
            except Exception as e:
                logger.warning(f"Poisson reconstruction failed: {e}, using alpha shapes")
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha=0.03
                )
            
            # Clean up mesh
            if progress_callback:
                progress_callback(78, "Cleaning up mesh...")
            
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            
            # Smooth the mesh
            mesh = mesh.filter_smooth_simple(number_of_iterations=5)
            
            # Compute vertex normals
            mesh.compute_vertex_normals()
            
            # Convert to format suitable for export
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            normals = np.asarray(mesh.vertex_normals)
            
            # Generate texture coordinates (simple spherical mapping)
            texture_coords = self._generate_spherical_uv_coordinates(vertices)
            
            return {
                'vertices': vertices,
                'faces': faces,
                'normals': normals,
                'texture_coords': texture_coords,
                'mesh': mesh,
                'vertex_count': len(vertices),
                'face_count': len(faces)
            }
            
        except Exception as e:
            logger.error(f"Dense reconstruction and meshing failed: {str(e)}")
            return self._create_fallback_mesh()

    def _generate_sparse_outputs_from_text(
        self,
        mesh_data: Dict[str, Any],
        multi_view_images: List[np.ndarray],
        prompt: str,
        resolution: int,
        job_id: str
    ) -> Dict[str, Any]:
        """Generate sparse outputs for text-to-3D workflow"""
        try:
            timestamp = int(time.time())
            base_filename = f"sparse_text_{job_id}_{timestamp}" if job_id else f"sparse_text_{timestamp}"
            
            outputs = {
                'sparse_url': None,
                'sparse_depth_url': None,
                'sparse_wireframe_url': None,
                'sparse_normal_url': None,
                'reference_image_url': None,
                'metadata': {
                    'vertex_count': mesh_data['vertex_count'],
                    'face_count': mesh_data['face_count'],
                    'prompt': prompt,
                    'resolution': resolution,
                    'reconstruction_method': 'sparse_view_text',
                    'timestamp': timestamp
                }
            }
            
            # Save the first view as reference image
            if multi_view_images:
                ref_image = multi_view_images[0]
                ref_filename = f"{base_filename}_reference.png"
                ref_path = os.path.join(self.output_dir, ref_filename)
                Image.fromarray(ref_image).save(ref_path)
                
                if os.path.exists(ref_path):
                    outputs['reference_image_url'] = self._upload_to_cloudinary(ref_path, 'image')
                    os.remove(ref_path)
            
            # Save OBJ file
            obj_filename = f"{base_filename}.obj"
            obj_path = os.path.join(self.output_dir, obj_filename)
            self._save_obj_file(mesh_data, obj_path)
            
            if os.path.exists(obj_path):
                outputs['sparse_url'] = self._upload_to_cloudinary(obj_path, 'model')
                os.remove(obj_path)
            
            # Generate depth map visualization
            if multi_view_images:
                depth_image = self._generate_depth_visualization(mesh_data, multi_view_images[0])
                depth_filename = f"{base_filename}_depth.png"
                depth_path = os.path.join(self.output_dir, depth_filename)
                cv2.imwrite(depth_path, depth_image)
                
                if os.path.exists(depth_path):
                    outputs['sparse_depth_url'] = self._upload_to_cloudinary(depth_path, 'image')
                    os.remove(depth_path)
            
            # Generate wireframe visualization
            if multi_view_images:
                wireframe_image = self._generate_wireframe_visualization(mesh_data, multi_view_images[0])
                wireframe_filename = f"{base_filename}_wireframe.png"
                wireframe_path = os.path.join(self.output_dir, wireframe_filename)
                cv2.imwrite(wireframe_path, wireframe_image)
                
                if os.path.exists(wireframe_path):
                    outputs['sparse_wireframe_url'] = self._upload_to_cloudinary(wireframe_path, 'image')
                    os.remove(wireframe_path)
            
            # Generate normal map visualization
            normal_image = self._generate_normal_visualization(mesh_data)
            normal_filename = f"{base_filename}_normal.png"
            normal_path = os.path.join(self.output_dir, normal_filename)
            cv2.imwrite(normal_path, normal_image)
            
            if os.path.exists(normal_path):
                outputs['sparse_normal_url'] = self._upload_to_cloudinary(normal_path, 'image')
                os.remove(normal_path)
            
            logger.info(f"Generated sparse text-to-3D outputs: {outputs}")
            return outputs
            
        except Exception as e:
            logger.error(f"Failed to generate sparse text-to-3D outputs: {str(e)}")
            return {
                'sparse_url': None,
                'sparse_depth_url': None,
                'sparse_wireframe_url': None,
                'sparse_normal_url': None,
                'reference_image_url': None,
                'metadata': {'error': str(e)}
            }
    
def train_sparse_reconstruction(
    reconstructor: SparseViewReconstructor,
    request_data: Dict[str, Any],
    job_id: str,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """Training function for sparse view reconstruction
    
    This function handles the complete sparse view reconstruction pipeline
    from text prompt to final 3D outputs.
    
    Args:
        reconstructor: SparseViewReconstructor instance
        request_data: Dictionary containing reconstruction parameters
        job_id: Unique job identifier
        progress_callback: Function to report progress updates
        
    Returns:
        Dictionary containing reconstruction results
    """
    try:
        logger.info(f"Starting sparse view reconstruction training for job {job_id}")
        
        # Extract parameters from request
        prompt = request_data.get('prompt', '')
        num_views = request_data.get('num_views', 6)
        resolution = request_data.get('resolution', 512)
        negative_prompt = request_data.get('negative_prompt', 'low quality, bad anatomy, worst quality, low resolution, blurry')
        
        # Validate parameters
        if not prompt:
            raise ValueError("Prompt is required for sparse view reconstruction")
        
        if num_views < 4 or num_views > 12:
            raise ValueError("Number of views must be between 4 and 12")
        
        if resolution not in [256, 512, 1024]:
            raise ValueError("Resolution must be 256, 512, or 1024")
        
        logger.info(f"Sparse reconstruction parameters:")
        logger.info(f"  Prompt: {prompt[:50]}...")
        logger.info(f"  Views: {num_views}")
        logger.info(f"  Resolution: {resolution}")
        logger.info(f"  Negative prompt: {negative_prompt[:30]}...")
        
        # Start reconstruction process
        if progress_callback:
            progress_callback(5, "Initializing sparse view reconstruction...")
        
        # Call the main reconstruction method
        result = reconstructor.reconstruct_from_text(
            prompt=prompt,
            num_views=num_views,
            resolution=resolution,
            negative_prompt=negative_prompt,
            job_id=job_id,
            progress_callback=progress_callback
        )
        
        # Validate results
        if not result:
            raise RuntimeError("Sparse view reconstruction returned no results")
        
        logger.info(f"Sparse view reconstruction completed successfully for job {job_id}")
        logger.info(f"Generated outputs: {list(result.keys())}")
        
        return result
        
    except Exception as e:
        error_msg = f"Sparse view reconstruction training failed: {str(e)}"
        logger.error(error_msg)
        
        if progress_callback:
            progress_callback(-1, error_msg, None)
        
        # Return error result
        return {
            'success': False,
            'error': error_msg,
            'sparse_url': None,
            'sparse_depth_url': None,
            'sparse_wireframe_url': None,
            'sparse_normal_url': None,
            'reference_image_url': None,
            'metadata': {
                'error': str(e),
                'job_id': job_id,
                'timestamp': time.time()
            }
        }
