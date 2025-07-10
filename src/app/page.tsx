'use client';

import { useState, useEffect, useRef } from 'react';
import { FaDownload, FaMagic, FaFileUpload, FaKeyboard } from 'react-icons/fa';
import toast from 'react-hot-toast';
import ImageUpload from '@/components/ImageUpload';

type GenerationMode = 'text' | 'upload';

export default function Home() {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedUrls, setGeneratedUrls] = useState<{
    image_url: string | null;
    model_url: string | null;
    depth_map_url: string | null;
    preview_mesh_url: string | null;    // Fast preview mesh
    nerf_weights_url: string | null;    // NeRF model weights (.pth/.ckpt)
    nerf_config_url: string | null;     // NeRF configuration (.json)
    nerf_mesh_url: string | null;       // High-quality mesh from NeRF (.obj)
    nerf_viewer_url: string | null;     // Interactive web viewer
  }>({
    image_url: null,
    model_url: null,
    depth_map_url: null,
    preview_mesh_url: null,
    nerf_weights_url: null,
    nerf_config_url: null,
    nerf_mesh_url: null,
    nerf_viewer_url: null
  });
  const [generationStep, setGenerationStep] = useState(0);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [generationMessage, setGenerationMessage] = useState('');
  const [mode, setMode] = useState<GenerationMode>('text');
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadMessage, setUploadMessage] = useState('');
  const [depthModel, setDepthModel] = useState<'intel' | 'apple'>('intel');
  const [generationMode, setGenerationMode] = useState<'fast' | 'premium' | 'both'>('fast');
  const [previewReady, setPreviewReady] = useState(false);
  const [premiumProcessing, setPremiumProcessing] = useState(false);

  // Refs to store abort controllers and job IDs for ongoing requests
  const uploadAbortControllerRef = useRef<AbortController | null>(null);
  const generateAbortControllerRef = useRef<AbortController | null>(null);
  const currentJobIdRef = useRef<string | null>(null);
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Function to start progress tracking
  const startProgressTracking = (jobId: string, isUpload: boolean = false, waitForCompletion: boolean = false) => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
    }
    
    progressIntervalRef.current = setInterval(async () => {
      try {
        const response = await fetch(`/api/python/progress/${jobId}`);
        if (response.ok) {
          const progressData = await response.json();
          
          if (isUpload) {
            setUploadProgress(progressData.progress || 0);
            setUploadMessage(progressData.message || '');
          } else {
            setGenerationProgress(progressData.progress || 0);
            setGenerationMessage(progressData.message || '');
          }
          
          // Handle completion for premium mode
          if (progressData.stage === 'completed') {
            clearInterval(progressIntervalRef.current!);
            progressIntervalRef.current = null;
            
            // If this is a premium NeRF job and has results, update the UI
            if (waitForCompletion && progressData.result) {
              const result = progressData.result;
              setGeneratedUrls(prev => ({
                ...prev,
                image_url: result.image_url || prev.image_url,
                nerf_weights_url: result.nerf_weights_url || null,
                nerf_config_url: result.nerf_config_url || null,
                nerf_mesh_url: result.nerf_mesh_url || null,
                nerf_viewer_url: result.nerf_viewer_url || null,
                model_url: result.nerf_mesh_url || result.model_url || prev.model_url
              }));
              
              setPreviewReady(true);
              setPremiumProcessing(false);
              toast.success('Premium NeRF model ready! ✨');
            }
          }
          
          // Stop tracking if failed
          if (progressData.stage === 'error') {
            clearInterval(progressIntervalRef.current!);
            progressIntervalRef.current = null;
            
            if (waitForCompletion) {
              setPremiumProcessing(false);
              toast.error('NeRF generation failed');
            }
          }
        }
      } catch (error) {
        // Silently fail - don't spam console with progress errors
      }
    }, 1000); // Check every second
  };

  // Function to stop progress tracking
  const stopProgressTracking = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  };

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      stopProgressTracking();
      
      // Abort any pending requests
      if (uploadAbortControllerRef.current) {
        uploadAbortControllerRef.current.abort();
      }
      if (generateAbortControllerRef.current) {
        generateAbortControllerRef.current.abort();
      }
      
      // Clean up object URLs
      if (uploadedImageUrl && uploadedImageUrl.startsWith('blob:')) {
        URL.revokeObjectURL(uploadedImageUrl);
      }
    };
  }, [uploadedImageUrl]);

  // Handle file selection (no upload yet)
  const handleFileSelect = (file: File) => {
    // Clean up previous object URL if exists
    if (uploadedImageUrl && uploadedImageUrl.startsWith('blob:')) {
      URL.revokeObjectURL(uploadedImageUrl);
    }
    
    setSelectedFile(file);
    const fileUrl = URL.createObjectURL(file);
    setUploadedImageUrl(fileUrl);
    
    // Clear previous results
    setGeneratedUrls({
      image_url: null,
      model_url: null,
      depth_map_url: null,
      preview_mesh_url: null,
      nerf_weights_url: null,
      nerf_config_url: null,
      nerf_mesh_url: null,
      nerf_viewer_url: null
    });
  };

  // Handle image conversion (separate from file selection)
  const handleConvertImage = async () => {
    if (!selectedFile) {
      toast.error('Please select an image first');
      return;
    }

    // Abort any previous upload request
    if (uploadAbortControllerRef.current) {
      uploadAbortControllerRef.current.abort();
    }

    // Create new abort controller for this request
    const abortController = new AbortController();
    uploadAbortControllerRef.current = abortController;
    
    setIsUploading(true);

    const uploadPromise = new Promise(async (resolve, reject) => {
      try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('depth_model', depthModel);

        const response = await fetch('/api/python/upload', {
          method: 'POST',
          body: formData,
          signal: abortController.signal, // Add abort signal
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Upload failed');
        }

        const data = await response.json();
        
        // Store job ID for potential cancellation and start progress tracking
        if (data.job_id) {
          currentJobIdRef.current = data.job_id;
          startProgressTracking(data.job_id, true);
        }
        
        // Update with server URLs (replace the local file URL)
        setGeneratedUrls({
          image_url: data.image_url,
          model_url: data.model_url,
          depth_map_url: data.depth_map_url || null,
          preview_mesh_url: data.preview_mesh_url || null,
          nerf_weights_url: data.nerf_weights_url || null,
          nerf_config_url: data.nerf_config_url || null,
          nerf_mesh_url: data.nerf_mesh_url || null,
          nerf_viewer_url: data.nerf_viewer_url || null
        });
        
        // Clean up the local object URL and update with server URL
        URL.revokeObjectURL(uploadedImageUrl);
        setUploadedImageUrl(data.image_url);

        resolve(data);
      } catch (error: any) {
        // Don't show error if request was aborted
        if (error.name === 'AbortError') {
          console.log('Upload request was aborted');
          return;
        }
        
        console.error('Error:', error);
        reject(error);
      } finally {
        // Clear the abort controller reference and job ID
        if (uploadAbortControllerRef.current === abortController) {
          uploadAbortControllerRef.current = null;
        }
        currentJobIdRef.current = null;
      }
    });

    toast.promise(uploadPromise, {
      loading: 'Converting your image to 3D...',
      success: 'Conversion complete!',
      error: (err) => {
        // Don't show error toast for aborted requests
        if (err?.name === 'AbortError') return null;
        
        if (err.message.includes('Python API is not running')) {
          return 'Server is not running. Please try again in a few minutes.';
        }
        if (err.message.includes('timeout')) {
          return 'Conversion took too long. Please try again.';
        }
        return err.message || 'Failed to convert. Please try again.';
      },
    });

    try {
      await uploadPromise;
    } catch (error: any) {
      // Error is handled by toast.promise, but check if it was aborted
      if (error?.name === 'AbortError') {
        console.log('Upload was aborted by user');
      }
    } finally {
      setIsUploading(false);
      stopProgressTracking();
    }
  };

  const handleRemoveImage = () => {
    // Clean up object URL if it's a local file URL
    if (uploadedImageUrl && uploadedImageUrl.startsWith('blob:')) {
      URL.revokeObjectURL(uploadedImageUrl);
    }
    setUploadedImageUrl('');
    setSelectedFile(null);
    setGeneratedUrls({
      image_url: null,
      model_url: null,
      depth_map_url: null,
      preview_mesh_url: null,
      nerf_weights_url: null,
      nerf_config_url: null,
      nerf_mesh_url: null,
      nerf_viewer_url: null
    });
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a prompt');
      return;
    }

    // Abort any previous generate request
    if (generateAbortControllerRef.current) {
      generateAbortControllerRef.current.abort();
    }

    // Create new abort controller for this request
    const abortController = new AbortController();
    generateAbortControllerRef.current = abortController;
    
    setIsGenerating(true);
    setGenerationStep(1);
    setPreviewReady(false);
    setPremiumProcessing(false);
    setGeneratedUrls({
      image_url: null,
      model_url: null,
      depth_map_url: null,
      preview_mesh_url: null,
      nerf_weights_url: null,
      nerf_config_url: null,
      nerf_mesh_url: null,
      nerf_viewer_url: null
    });
    
    const generatePromise = new Promise(async (resolve, reject) => {
      try {
        // Different workflows based on generation mode
        if (generationMode === 'premium') {
          // Premium mode: Generate NeRF directly (no fast preview)
          setPremiumProcessing(true);
          
          const response = await fetch('/api/python/generate', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
              prompt, 
              depth_model: depthModel,
              mode: 'premium'
            }),
            signal: abortController.signal,
          });
          
          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Premium generation failed');
          }
          
          const data = await response.json();
          
          // Store job ID for tracking
          if (data.job_id) {
            currentJobIdRef.current = data.job_id;
            startProgressTracking(data.job_id, false, true); // Wait for completion
          }
          
          resolve(data);
          
        } else {
          // Fast mode OR Both mode: Generate fast preview first
          const fastResponse = await fetch('/api/python/generate', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
              prompt, 
              depth_model: depthModel,
              mode: 'fast'
            }),
            signal: abortController.signal,
          });
          
          if (!fastResponse.ok) {
            const errorData = await fastResponse.json();
            throw new Error(errorData.error || 'Fast generation failed');
          }
          
          const fastData = await fastResponse.json();
          
          // Store job ID for tracking
          if (fastData.job_id) {
            currentJobIdRef.current = fastData.job_id;
            startProgressTracking(fastData.job_id, false);
          }
          
          // Update with fast preview results
          setGeneratedUrls(prev => ({
            ...prev,
            image_url: fastData.image_url,
            depth_map_url: fastData.depth_map_url || null,
            preview_mesh_url: fastData.model_url, // Fast mesh
            model_url: fastData.model_url // Default to fast mesh
          }));
          
          setPreviewReady(true);
          toast.success('Preview ready! 🚀');
          
          // If "both" mode, start NeRF generation in background
          if (generationMode === 'both') {
            setPremiumProcessing(true);
            
            const premiumResponse = await fetch('/api/python/generate-nerf', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ 
                prompt,
                image_url: fastData.image_url, // Use generated image as input
                depth_model: depthModel,
                steps: 3000, // Optimized steps for speed/quality balance
                resolution: 512 // Reasonable resolution
              }),
              signal: abortController.signal,
            });

            if (premiumResponse.ok) {
              const premiumData = await premiumResponse.json();
              
              // Start progress tracking for NeRF
              if (premiumData.job_id) {
                currentJobIdRef.current = premiumData.job_id;
                startProgressTracking(premiumData.job_id, false);
              }
            }
          }
          
          resolve(fastData);
        }
      } catch (error: any) {
        // Don't show error if request was aborted
        if (error.name === 'AbortError') {
          console.log('Generate request was aborted');
          return;
        }
        
        console.error('Error:', error);
        reject(error);
      } finally {
        // Clear the abort controller reference and job ID
        if (generateAbortControllerRef.current === abortController) {
          generateAbortControllerRef.current = null;
        }
        currentJobIdRef.current = null;
      }
    });
    
    const loadingMessage = generationMode === 'premium' 
      ? 'Training NeRF model...' 
      : generationMode === 'both'
      ? 'Creating preview + NeRF...'
      : 'Generating your image...';
    
    toast.promise(generatePromise, {
      loading: loadingMessage,
      success: (data: any) => {
        return generationMode === 'premium' 
          ? 'NeRF model training complete! 🎉'
          : data.source === 'fallback' 
          ? 'Image created with fallback mode' 
          : generationMode === 'both'
          ? 'Preview ready, NeRF processing...'
          : 'Generation complete!';
      },
      error: (err) => {
        // Don't show error toast for aborted requests
        if (err?.name === 'AbortError') return null;
        
        if (err.message.includes('Python API is not running')) {
          return 'Server is not running. Please try again in a few minutes.';
        }
        if (err.message.includes('timeout')) {
          return 'Generation took too long. Please try again.';
        }
        return err.message || 'Failed to generate. Please try again.';
      },
    });
    
    try {
      await generatePromise;
    } catch (error: any) {
      // Error is handled by toast.promise, but check if it was aborted
      if (error?.name === 'AbortError') {
        console.log('Generation was aborted by user');
      }
    } finally {
      setIsGenerating(false);
      setPremiumProcessing(false);
      stopProgressTracking();
    }
  };

  // Download all available files function
  const downloadAll = async () => {
    const downloads = [];
    const timestamp = Date.now();
    
    // Collect all available downloads
    if (generatedUrls.image_url) {
      downloads.push({
        url: generatedUrls.image_url,
        filename: `3dify-image-${timestamp}.png`
      });
    }
    
    if (generatedUrls.depth_map_url) {
      downloads.push({
        url: generatedUrls.depth_map_url,
        filename: `3dify-depth-${timestamp}.png`
      });
    }
    
    if (generatedUrls.preview_mesh_url) {
      downloads.push({
        url: generatedUrls.preview_mesh_url,
        filename: `3dify-preview-mesh-${timestamp}.obj`
      });
    }
    
    if (generatedUrls.nerf_weights_url) {
      downloads.push({
        url: generatedUrls.nerf_weights_url,
        filename: `3dify-nerf-weights-${timestamp}.pth`
      });
    }
    
    if (generatedUrls.nerf_mesh_url) {
      downloads.push({
        url: generatedUrls.nerf_mesh_url,
        filename: `3dify-nerf-mesh-${timestamp}.obj`
      });
    }
    
    if (generatedUrls.nerf_config_url) {
      downloads.push({
        url: generatedUrls.nerf_config_url,
        filename: `3dify-nerf-config-${timestamp}.json`
      });
    }
    
    // Download all files with delay between downloads
    if (downloads.length > 0) {
      toast.promise(
        Promise.all(downloads.map(async (download, index) => {
          // Add delay between downloads to avoid overwhelming the browser
          await new Promise(resolve => setTimeout(resolve, index * 500));
          
          const link = document.createElement('a');
          link.href = download.url;
          link.download = download.filename;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        })),
        {
          loading: `Downloading ${downloads.length} files...`,
          success: `Successfully downloaded ${downloads.length} files!`,
          error: 'Failed to download some files'
        }
      );
    }
  };

  return (
    <main className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-10 backdrop-blur-md border-b border-white/5 bg-background/60">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <a href="/" className="text-2xl font-bold text-gradient">
            3Dify
          </a>
          
          <div className="flex items-center space-x-4">
            <a 
              href="/about" 
              className="text-text-secondary hover:text-primary transition-colors"
              title="About"
            >
              About
            </a>
            <a 
              href="https://github.com/Aniket-404/text-to-3d-nextjs" 
              target="_blank"
              className="text-text-secondary hover:text-primary transition-colors"
              title="GitHub"
            >
              GitHub
            </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-grow container mx-auto px-4 py-10">
        <div className="split-layout">
          {/* Left Panel - Input */}
          <div className="glass-panel p-6">
            <h1 className="text-3xl font-bold mb-6 text-gradient">
              Create Your 3D Model
            </h1>
            
            {/* Mode Switch */}
            <div className="flex mb-6 bg-surface/30 rounded-lg p-1">
              <button
                onClick={() => setMode('text')}
                className={`flex-1 flex items-center justify-center space-x-2 px-4 py-2 rounded-md transition-colors ${
                  mode === 'text' 
                    ? 'bg-primary text-white' 
                    : 'text-text-secondary hover:text-primary'
                }`}
              >
                <FaKeyboard />
                <span>Text to 3D</span>
              </button>
              <button
                onClick={() => setMode('upload')}
                className={`flex-1 flex items-center justify-center space-x-2 px-4 py-2 rounded-md transition-colors ${
                  mode === 'upload' 
                    ? 'bg-primary text-white' 
                    : 'text-text-secondary hover:text-primary'
                }`}
              >
                <FaFileUpload />
                <span>Image to 3D</span>
              </button>
            </div>
            
            {/* Depth Model Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-text-secondary mb-2">
                Depth Estimation Model
              </label>
              <select
                value={depthModel}
                onChange={(e) => setDepthModel(e.target.value as 'intel' | 'apple')}
                disabled={isGenerating || isUploading}
                className="w-full p-3 rounded-lg input-gradient focus:outline-none text-text-primary bg-surface border border-white/10 focus:border-primary/50"
              >
                <option value="intel">Intel DPT BEIT Large 512</option>
                <option value="apple">Apple DepthPro (Experimental)</option>
              </select>
              <p className="text-xs text-text-secondary mt-1">
                {depthModel === 'intel' 
                  ? 'Intel DPT BEIT Large 512 - Fast and reliable depth estimation'
                  : 'Apple DepthPro - High-quality depth estimation (currently using Intel DPT as fallback)'
                }
              </p>
            </div>
            
            {/* Generation Quality Mode */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-text-secondary mb-2">
                Quality Mode
              </label>
              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => setGenerationMode('fast')}
                  disabled={isGenerating || isUploading}
                  className={`p-3 rounded-lg border transition-colors text-sm ${
                    generationMode === 'fast'
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-white/10 bg-surface/30 hover:border-white/20'
                  } ${(isGenerating || isUploading) ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <div className="font-medium">Fast</div>
                  <div className="text-xs text-text-secondary">5-10s</div>
                </button>
                <button
                  onClick={() => setGenerationMode('premium')}
                  disabled={isGenerating || isUploading}
                  className={`p-3 rounded-lg border transition-colors text-sm ${
                    generationMode === 'premium'
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-white/10 bg-surface/30 hover:border-white/20'
                  } ${(isGenerating || isUploading) ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <div className="font-medium">Premium</div>
                  <div className="text-xs text-text-secondary">2-5min</div>
                </button>
                <button
                  onClick={() => setGenerationMode('both')}
                  disabled={isGenerating || isUploading}
                  className={`p-3 rounded-lg border transition-colors text-sm ${
                    generationMode === 'both'
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-white/10 bg-surface/30 hover:border-white/20'
                  } ${(isGenerating || isUploading) ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <div className="font-medium">Both</div>
                  <div className="text-xs text-text-secondary">Fast + NeRF</div>
                </button>
              </div>
              <p className="text-xs text-text-secondary mt-2">
                {generationMode === 'fast' && 'Fast depth-based mesh generation'}
                {generationMode === 'premium' && 'High-quality NeRF with downloadable weights, config, and mesh'}
                {generationMode === 'both' && 'Get fast preview + premium NeRF model (recommended)'}
              </p>
            </div>
            
            {/* Content based on mode */}
            {mode === 'text' ? (
              <div className="space-y-4">
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="A futuristic space rover with robotic arms..."
                  className="w-full h-40 p-4 rounded-lg input-gradient focus:outline-none resize-none"
                  disabled={isGenerating}
                />
                
                <button
                  onClick={handleGenerate}
                  disabled={isGenerating || !prompt.trim()}
                  className={`w-full gradient-button py-3 flex items-center justify-center space-x-2 ${
                    isGenerating || !prompt.trim() ? 'opacity-50 cursor-not-allowed' : 'hover:shadow-glow'
                  }`}
                >
                  <FaMagic className={isGenerating ? 'animate-spin' : ''} />
                  <span>
                    {isGenerating ? 'Generating...' : 'Generate 3D Model'}
                  </span>
                </button>
                
                {isGenerating && (
                  <div className="mt-4">
                    <div className="w-full bg-surface/30 h-2 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-primary transition-all duration-300"
                        style={{ width: `${generationProgress}%` }}
                      />
                    </div>
                    <div className="text-sm text-text-secondary mt-2">
                      {generationMessage || 'Generating your 3D model...'}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                <ImageUpload
                  onImageUpload={handleFileSelect}
                  isUploading={isUploading}
                  uploadedImageUrl={uploadedImageUrl}
                  onRemoveImage={handleRemoveImage}
                />
                
                {selectedFile && !isUploading && (
                  <button
                    onClick={handleConvertImage}
                    disabled={!selectedFile}
                    className={`w-full gradient-button py-3 flex items-center justify-center space-x-2 ${
                      !selectedFile ? 'opacity-50 cursor-not-allowed' : 'hover:shadow-glow'
                    }`}
                  >
                    <FaMagic />
                    <span>Convert to 3D Model</span>
                  </button>
                )}
                
                {isUploading && (
                  <div className="mt-4">
                    <div className="w-full bg-surface/30 h-2 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-primary transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                    <div className="text-sm text-text-secondary mt-2">
                      {uploadMessage || 'Uploading and converting to 3D model...'}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
          
          {/* Right Panel - Output */}
          <div className="glass-panel p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-bold">
                {mode === 'text' ? 'Generated Results' : 'Conversion Results'}
              </h2>
              <p className="text-text-secondary text-sm mt-1">
                {mode === 'text' 
                  ? 'Your AI-generated image, depth map, and 3D model will appear here'
                  : 'Your uploaded image, depth map, and converted 3D model will appear here'
                }
              </p>
            </div>
            <div className="space-y-6">
              {/* Top Row: Image and Depth Map side by side */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Image Preview */}
                <div className="aspect-square rounded-lg overflow-hidden">
                  {generatedUrls.image_url ? (
                    <div className="relative w-full h-full">
                      <img 
                        src={generatedUrls.image_url} 
                        alt={mode === 'text' ? (prompt || 'Generated image') : 'Uploaded image'} 
                        className="w-full h-full object-cover rounded-lg"
                      />
                      <div className="absolute bottom-2 left-2">
                        <a
                          href={generatedUrls.image_url}
                          download={`3dify-image-${Date.now()}.png`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center space-x-2 bg-surface/80 hover:bg-surface px-3 py-2 rounded-md transition-colors"
                        >
                          <FaDownload />
                          <span>Download Image</span>
                        </a>
                      </div>
                      <div className="absolute top-2 left-2 bg-black/70 text-white px-2 py-1 rounded text-xs">
                        {mode === 'text' ? 'Generated Image' : 'Uploaded Image'}
                      </div>
                    </div>
                  ) : (
                    <div className="w-full h-full flex items-center justify-center bg-black/40 rounded-lg">
                      {(isGenerating || isUploading) ? (
                        <div className="text-center">
                          <div className="inline-block p-3 rounded-full bg-surface/30 mb-4">
                            <FaMagic className="text-3xl animate-spin text-primary" />
                          </div>
                          <p>{mode === 'text' ? 'Creating your image...' : 'Processing your image...'}</p>
                        </div>
                      ) : (
                        <div className="text-center p-6">
                          <p className="text-lg mb-2">
                            {mode === 'text' ? 'Your generated image will appear here' : 'Your uploaded image will appear here'}
                          </p>
                          <p className="text-sm text-text-secondary">
                            {mode === 'text' 
                              ? 'Enter a prompt and click Generate to start'
                              : 'Upload an image to start the conversion'
                            }
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
                
                {/* Depth Map Preview */}
                <div className="aspect-square rounded-lg overflow-hidden">
                  {generatedUrls.depth_map_url ? (
                    <div className="relative w-full h-full">
                      <img 
                        src={generatedUrls.depth_map_url} 
                        alt="Depth map" 
                        className="w-full h-full object-cover rounded-lg"
                      />
                      <div className="absolute bottom-2 left-2">
                        <a
                          href={generatedUrls.depth_map_url}
                          download={`3dify-depth-${Date.now()}.png`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center space-x-2 bg-surface/80 hover:bg-surface px-3 py-2 rounded-md transition-colors"
                        >
                          <FaDownload />
                          <span>Download Depth</span>
                        </a>
                      </div>
                      <div className="absolute top-2 left-2 bg-black/70 text-white px-2 py-1 rounded text-xs">
                        {depthModel === 'intel' ? 'Intel DPT' : 'Apple DepthPro'}
                      </div>
                    </div>
                  ) : (
                    <div className="w-full h-full flex items-center justify-center bg-black/40 rounded-lg">
                      {(isGenerating || isUploading) ? (
                        <div className="text-center">
                          <div className="inline-block p-3 rounded-full bg-surface/30 mb-4">
                            <FaMagic className="text-3xl animate-spin text-primary" />
                          </div>
                          <p>Generating depth map...</p>
                        </div>
                      ) : (
                        <div className="text-center p-6">
                          <p className="text-lg mb-2">Depth Map</p>
                          <p className="text-sm text-text-secondary">
                            The depth estimation will appear here
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
              
              {/* Bottom Row: Enhanced Downloads */}
              <div className="flex justify-center">
                <div className="w-full max-w-md space-y-4">
                  
                  {/* Fast Preview Download */}
                  {(generatedUrls.model_url || generatedUrls.preview_mesh_url) && (
                    <div className="glass-panel p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <div className="font-medium text-sm">Standard 3D Mesh</div>
                          <div className="text-xs text-text-secondary">OBJ format - Fast generation</div>
                        </div>
                        <div className="bg-green-500/20 text-green-400 px-2 py-1 rounded text-xs">
                          Ready
                        </div>
                      </div>
                      <a
                        href={generatedUrls.model_url || generatedUrls.preview_mesh_url || ''}
                        download={`3dify-mesh-${Date.now()}.obj`}
                        className="flex items-center justify-center space-x-2 bg-surface hover:bg-surface/80 px-4 py-2 rounded-md transition-colors w-full text-sm"
                      >
                        <FaDownload />
                        <span>Download Mesh (.obj)</span>
                      </a>
                    </div>
                  )}

                  {/* Premium Processing Status */}
                  {premiumProcessing && (
                    <div className="glass-panel p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <div className="font-medium text-sm">Premium NeRF Model</div>
                          <div className="text-xs text-text-secondary">High-quality neural radiance field</div>
                        </div>
                        <div className="bg-yellow-500/20 text-yellow-400 px-2 py-1 rounded text-xs animate-pulse">
                          Processing
                        </div>
                      </div>
                      <div className="w-full bg-surface/30 h-2 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-primary transition-all duration-300"
                          style={{ width: `${generationProgress}%` }}
                        />
                      </div>
                      <div className="text-xs text-text-secondary mt-2">
                        {generationMessage || 'Training neural radiance field...'}
                      </div>
                    </div>
                  )}

                  {/* NeRF Model Downloads */}
                  {(generatedUrls.nerf_weights_url || generatedUrls.nerf_mesh_url || generatedUrls.nerf_viewer_url) && (
                    <div className="glass-panel p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <div className="font-medium text-sm">NeRF Model Package</div>
                          <div className="text-xs text-text-secondary">Neural Radiance Field - Premium Quality</div>
                        </div>
                        <div className="bg-purple-500/20 text-purple-400 px-2 py-1 rounded text-xs">
                          Premium ✨
                        </div>
                      </div>
                      
                      <div className="space-y-2">
                        {/* NeRF Weights Download */}
                        {generatedUrls.nerf_weights_url && (
                          <a
                            href={generatedUrls.nerf_weights_url}
                            download={`3dify-nerf-weights-${Date.now()}.pth`}
                            className="flex items-center justify-center space-x-2 bg-primary hover:bg-primary/90 text-black px-4 py-2 rounded-md transition-colors w-full text-sm font-medium"
                          >
                            <FaDownload />
                            <span>Download NeRF Weights (.pth)</span>
                          </a>
                        )}
                        
                        {/* High-Quality Mesh from NeRF */}
                        {generatedUrls.nerf_mesh_url && (
                          <a
                            href={generatedUrls.nerf_mesh_url}
                            download={`3dify-nerf-mesh-${Date.now()}.obj`}
                            className="flex items-center justify-center space-x-2 bg-surface hover:bg-surface/80 px-4 py-2 rounded-md transition-colors w-full text-sm"
                          >
                            <FaDownload />
                            <span>Download Premium Mesh (.obj)</span>
                          </a>
                        )}
                        
                        {/* NeRF Configuration */}
                        {generatedUrls.nerf_config_url && (
                          <a
                            href={generatedUrls.nerf_config_url}
                            download={`3dify-nerf-config-${Date.now()}.json`}
                            className="flex items-center justify-center space-x-2 bg-surface/50 hover:bg-surface/70 px-4 py-2 rounded-md transition-colors w-full text-sm"
                          >
                            <FaDownload />
                            <span>Download Config (.json)</span>
                          </a>
                        )}
                        
                        {/* Interactive Viewer */}
                        {generatedUrls.nerf_viewer_url && (
                          <a
                            href={generatedUrls.nerf_viewer_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center justify-center space-x-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-md transition-colors w-full text-sm"
                          >
                            <span>🎮</span>
                            <span>Open Interactive Viewer</span>
                          </a>
                        )}
                      </div>
                      
                      {/* NeRF Info */}
                      <div className="mt-3 p-2 bg-surface/30 rounded text-xs text-text-secondary">
                        💡 NeRF weights can be loaded in Blender, Unity, or custom viewers
                      </div>
                    </div>
                  )}

                  {/* Download All Button */}
                  {(generatedUrls.model_url || generatedUrls.preview_mesh_url || generatedUrls.nerf_weights_url) && (
                    <button
                      onClick={downloadAll}
                      className="w-full flex items-center justify-center space-x-2 bg-gradient-to-r from-primary to-blue-500 hover:from-primary/90 hover:to-blue-600 text-black px-4 py-3 rounded-md transition-colors font-medium"
                    >
                      <FaDownload />
                      <span>Download All Files</span>
                    </button>
                  )}

                  {/* Fallback - No Models */}
                  {!generatedUrls.model_url && !generatedUrls.preview_mesh_url && !generatedUrls.nerf_weights_url && (
                    <div className="glass-panel p-6 text-center">
                      {(isGenerating || isUploading) ? (
                        <div className="text-center">
                          <div className="inline-block p-3 rounded-full bg-surface/30 mb-4">
                            <FaMagic className="text-3xl animate-spin text-primary" />
                          </div>
                          <p>
                            {generationMode === 'premium' ? 'Training NeRF model...' :
                             generationMode === 'both' ? 'Creating models...' :
                             mode === 'text' ? 'Creating 3D model...' : 'Processing your image...'}
                          </p>
                          {premiumProcessing && (
                            <p className="text-sm text-text-secondary mt-2">
                              Preview ready, NeRF training in progress...
                            </p>
                          )}
                        </div>
                      ) : (
                        <div className="text-center">
                          <p className="text-lg mb-2">3D model will appear here</p>
                          <p className="text-sm text-text-secondary">
                            {mode === 'text' 
                              ? 'Enter a prompt and click Generate to start'
                              : 'Upload an image to start the conversion'
                            }
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <footer className="border-t border-white/5 py-6">
        <div className="container mx-auto px-4 text-center text-text-secondary text-sm">
          <p>© 2025 3Dify - Text to 3D Model Converter</p>
        </div>
      </footer>
    </main>
  );
}
