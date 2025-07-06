'use client';

import { useState, useEffect, useRef } from 'react';
import { FaDownload, FaMagic, FaFileUpload, FaKeyboard } from 'react-icons/fa';
import toast from 'react-hot-toast';
import useAuth from '@/hooks/useAuth';
import { getFirebaseFirestore } from '@/lib/firebase';
import { collection, addDoc, serverTimestamp } from 'firebase/firestore';
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
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadMessage, setUploadMessage] = useState('');
  const { user } = useAuth();

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
          
          // Stop tracking when completed or error
          if (progressData.stage === 'completed' || progressData.stage === 'error') {
            clearInterval(progressIntervalRef.current!);
            progressIntervalRef.current = null;
          }
        }
      } catch (error) {
        console.error('Progress tracking error:', error);
      }
    }, 500); // Update every 500ms
  };

  // Function to stop progress tracking
  const stopProgressTracking = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  };

  // Function to cancel a backend job
  const cancelBackendJob = async (jobId: string) => {
    try {
      await fetch('/api/python/cancel', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ job_id: jobId }),
      });
      console.log(`Cancelled backend job: ${jobId}`);
    } catch (error) {
      console.error('Error cancelling backend job:', error);
    }
  };

  // Cleanup function to abort all ongoing requests
  const abortAllRequests = () => {
    if (uploadAbortControllerRef.current) {
      uploadAbortControllerRef.current.abort();
      uploadAbortControllerRef.current = null;
    }
    if (generateAbortControllerRef.current) {
      generateAbortControllerRef.current.abort();
      generateAbortControllerRef.current = null;
    }
    
    // Cancel backend job if one is running
    if (currentJobIdRef.current) {
      cancelBackendJob(currentJobIdRef.current);
      currentJobIdRef.current = null;
    }
    
    // Stop progress tracking
    stopProgressTracking();
  };

  // Effect to handle page unload/refresh
  useEffect(() => {
    const handleBeforeUnload = () => {
      abortAllRequests();
    };

    // Only add beforeunload listener, not visibilitychange
    // visibilitychange is too aggressive and triggers on tab switches
    window.addEventListener('beforeunload', handleBeforeUnload);

    // Cleanup on component unmount
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      abortAllRequests();
    };
  }, []);

  const handleImageUpload = async (file: File) => {
    // Abort any previous upload request
    if (uploadAbortControllerRef.current) {
      uploadAbortControllerRef.current.abort();
    }

    // Create new abort controller for this request
    const abortController = new AbortController();
    uploadAbortControllerRef.current = abortController;

    // Show the uploaded image immediately
    const fileUrl = URL.createObjectURL(file);
    setUploadedImageUrl(fileUrl);
    setGeneratedUrls({
      image_url: fileUrl, // Show immediately
      model_url: null,
      depth_map_url: null
    });

    setIsUploading(true);

    const uploadPromise = new Promise(async (resolve, reject) => {
      try {
        const formData = new FormData();
        formData.append('file', file);

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
        
        setUploadedImageUrl(data.image_url);

        // Clean up the local object URL
        URL.revokeObjectURL(fileUrl);

        // Save to Firestore if user is logged in
        if (user && data.success) {
          try {
            const db = getFirebaseFirestore();
            if (db) {
              await addDoc(collection(db, 'generations'), {
                userId: user.uid,
                type: 'upload',
                originalImageUrl: data.image_url,
                modelUrl: data.model_url,
                depthMapUrl: data.depth_map_url,
                fileName: file.name,
                createdAt: serverTimestamp(),
              });
            }
          } catch (error) {
            console.error('Error saving to Firestore:', error);
          }
        }

        resolve(data);
      } catch (error: any) {
        // Clean up the local object URL on error
        URL.revokeObjectURL(fileUrl);
        
        // Don't show error if request was aborted
        if (error.name === 'AbortError') {
          console.log('Upload request was aborted');
          return;
        }
        
        console.error('Upload Error:', error);
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
      loading: 'Uploading and converting to 3D...',
      success: 'Upload and conversion complete!',
      error: (err) => {
        // Don't show error toast for aborted requests
        if (err?.name === 'AbortError') return null;
        
        if (err.message.includes('Invalid file type')) {
          return 'Please upload a valid image file (JPEG, PNG, GIF, BMP, WebP)';
        }
        if (err.message.includes('File size too large')) {
          return 'File size too large. Maximum size is 10MB.';
        }
        return err.message || 'Upload failed. Please try again.';
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
          body: JSON.stringify({ prompt }),
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
        
        // Save to Firestore if user is logged in
        if (user && data.success) {
          try {
            const db = getFirebaseFirestore();
            if (db) {
              await addDoc(collection(db, 'generations'), {
                userId: user.uid,
                type: 'prompt',
                prompt: prompt,
                imageUrl: data.image_url,
                modelUrl: data.model_url,
                depthMapUrl: data.depth_map_url,
                createdAt: serverTimestamp(),
              });
            }
          } catch (error) {
            console.error('Error saving to Firestore:', error);
          }
        }
        
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
              href="#" 
              className="text-text-secondary hover:text-primary transition-colors"
              title="About"
            >
              About
            </a>
            <a 
              href="https://github.com" 
              target="_blank"
              className="text-text-secondary hover:text-primary transition-colors"
              title="GitHub"
            >
              GitHub
            </a>
            {user ? (
              <a 
                href="/dashboard" 
                className="flex items-center space-x-2 bg-surface/50 hover:bg-surface/80 px-3 py-2 rounded-md transition-colors"
              >
                <FaMagic />
                <span>Dashboard</span>
              </a>
            ) : (
              <a 
                href="/auth/login" 
                className="flex items-center space-x-2 bg-surface/50 hover:bg-surface/80 px-3 py-2 rounded-md transition-colors"
              >
                <FaMagic />
                <span>Login</span>
              </a>
            )}
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
                  onImageUpload={handleImageUpload}
                  isUploading={isUploading}
                  uploadedImageUrl={uploadedImageUrl}
                  onRemoveImage={handleRemoveImage}
                />
                
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
                  ? 'Your AI-generated image and 3D model will appear here'
                  : 'Your uploaded image and converted 3D model will appear here'
                }
              </p>
            </div>
            <div className="grid grid-cols-2 gap-4">
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
              
              {/* 3D Model Download */}
              <div className="aspect-square rounded-lg overflow-hidden flex flex-col items-center justify-center bg-black/10">
                {generatedUrls.model_url ? (
                  <>
                    <div className="text-center mb-4">
                      <span className="block text-lg font-semibold mb-2">3D Model (OBJ)</span>
                      <span className="text-sm text-text-secondary">Download the generated 3D model</span>
                    </div>
                    <a
                      href={generatedUrls.model_url}
                      download
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center space-x-2 bg-surface/80 hover:bg-surface px-3 py-2 rounded-md transition-colors"
                    >
                      <FaDownload />
                      <span>Download 3D Model</span>
                    </a>
                  </>
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    {(isGenerating || isUploading) ? (
                      <div className="text-center">
                        <div className="inline-block p-3 rounded-full bg-surface/30 mb-4">
                          <FaMagic className="text-3xl animate-spin text-primary" />
                        </div>
                        <p>Creating 3D model...</p>
                      </div>
                    ) : (
                      <div className="text-center p-6">
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
      
      {/* Footer */}
      <footer className="border-t border-white/5 py-6">
        <div className="container mx-auto px-4 text-center text-text-secondary text-sm">
          <p>© 2025 3Dify - Text to 3D Model Converter</p>
          <p className="mt-1">
            Powered by Next.js, Three.js, and AI
          </p>
        </div>
      </footer>
    </main>
  );
}
