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
  }>({
    image_url: null,
    model_url: null,
    depth_map_url: null
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

  // Refs to store abort controllers and job IDs for ongoing requests
  const uploadAbortControllerRef = useRef<AbortController | null>(null);
  const generateAbortControllerRef = useRef<AbortController | null>(null);
  const currentJobIdRef = useRef<string | null>(null);
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Function to start progress tracking
  const startProgressTracking = (jobId: string, isUpload: boolean = false) => {
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
          
          // Stop tracking if completed or failed
          if (progressData.stage === 'completed' || progressData.stage === 'error') {
            clearInterval(progressIntervalRef.current!);
            progressIntervalRef.current = null;
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
      depth_map_url: null
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
          depth_map_url: data.depth_map_url || null
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
      depth_map_url: null
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
    setGeneratedUrls({
      image_url: null,
      model_url: null,
      depth_map_url: null
    });
    
    const generatePromise = new Promise(async (resolve, reject) => {
      try {
        const response = await fetch('/api/python/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ prompt, depth_model: depthModel }),
          signal: abortController.signal, // Add abort signal
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Generation failed');
        }
        
        const data = await response.json();
        
        // Store job ID for potential cancellation and start progress tracking
        if (data.job_id) {
          currentJobIdRef.current = data.job_id;
          startProgressTracking(data.job_id, false);
        }
        
        setGeneratedUrls({
          image_url: data.image_url,
          model_url: data.model_url,
          depth_map_url: data.depth_map_url || null
        });
        
        resolve(data);
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
    
    toast.promise(generatePromise, {
      loading: 'Generating your image...',
      success: (data: any) => {
        return data.source === 'fallback' 
          ? 'Image created with fallback mode' 
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
      stopProgressTracking();
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
              
              {/* Bottom Row: 3D Model Download */}
              <div className="flex justify-center">
                <div className="w-full max-w-md">
                  {generatedUrls.model_url ? (
                    <div className="glass-panel p-6 text-center">
                      <div className="inline-block p-4 rounded-full bg-primary/20 mb-4">
                        <FaDownload className="text-3xl text-primary" />
                      </div>
                      <div className="mb-6">
                        <span className="block text-lg font-semibold mb-2">3D Model Ready</span>
                        <span className="text-sm text-text-secondary">OBJ format - Ready for download</span>
                      </div>
                      <a
                        href={generatedUrls.model_url}
                        download={`3dify-model-${Date.now()}.obj`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center justify-center space-x-2 bg-primary hover:bg-primary/90 text-black px-6 py-3 rounded-md transition-colors font-medium w-full"
                      >
                        <FaDownload />
                        <span>Download 3D Model</span>
                      </a>
                    </div>
                  ) : (
                    <div className="glass-panel p-6 text-center">
                      {(isGenerating || isUploading) ? (
                        <div className="text-center">
                          <div className="inline-block p-3 rounded-full bg-surface/30 mb-4">
                            <FaMagic className="text-3xl animate-spin text-primary" />
                          </div>
                          <p>Creating 3D model...</p>
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
