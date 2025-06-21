'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { FaUser, FaMagic, FaInfoCircle, FaGithub, FaDownload, FaShare } from 'react-icons/fa';
import ModelViewer from '@/components/ModelViewer';
import toast from 'react-hot-toast';
import useAuth from '@/hooks/useAuth';
import { getFirebaseFirestore } from '@/lib/firebase';
import { collection, addDoc, serverTimestamp } from 'firebase/firestore';
import { generateCloudinaryPaths } from '@/lib/cloudinaryUtils';

export default function Home() {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [modelUrl, setModelUrl] = useState<string | null>(null);
  const [generationStep, setGenerationStep] = useState(0);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [depthUrl, setDepthUrl] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'image' | '3d'>('image');
  const { user } = useAuth();

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a prompt');
      return;
    }
    
    setIsGenerating(true);
    setGenerationStep(1);
    setModelUrl(null);
    setImageUrl(null);
    setDepthUrl(null);
    setViewMode('image');
    
    const generatePromise = new Promise(async (resolve, reject) => {
      try {
        // Step 1: Generate the image and depth map
        const response = await fetch('/api/python/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ prompt }),
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          console.error('API Error:', errorData);
          throw new Error(errorData.error || 'Failed to generate 3D model');
        }
        
        const data = await response.json();
        console.log('API Response:', data);
          // Set the image URL if it exists
        if (data.url) {
          setImageUrl(data.url);
          setGenerationStep(3);
          // Check if it's a fallback image
          if (data.source === 'fallback') {
            toast('Using a fallback image due to generation issues. You can still view it as a basic 3D model.', {
              icon: '⚠️',
              duration: 5000
            });
          }
        } else {
          throw new Error('No image URL returned from API');
        }
        
        // If we have a valid model URL, set it
        if (data.publicId) {
          setModelUrl(data.publicId);
          setGenerationStep(4);
        }
        
        // Store the generated model if user is logged in
        if (user) {
          try {
            const db = getFirebaseFirestore();
            if (db) {
              const modelsCollection = collection(db, 'models');
              await addDoc(modelsCollection, {
                userId: user.uid,
                prompt,
                imagePath: data.imageUrl,
                publicId: data.publicId,
                createdAt: serverTimestamp()
              });
            }
          } catch (err) {
            console.error('Error saving model:', err);
            // Don't throw here, as the generation was successful
            toast.error('Failed to save to your account, but generation was successful');
          }
        }
        
        resolve(data);
      } catch (error: any) {
        console.error('Error:', error);
        reject(error);
      }
    });    toast.promise(generatePromise, {
      loading: 'Generating your image...',
      success: (data: any) => {
        // Check source to provide the right success message
        return data.source === 'fallback' 
          ? 'Image created with fallback mode' 
          : 'Generation complete!';
      },
      error: (err) => {
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
    } catch (error) {
      // Error is handled by toast.promise
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-10 backdrop-blur-md border-b border-white/5 bg-background/60">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <Link href="/" className="text-2xl font-bold text-gradient">
            3Dify
          </Link>
          
          <div className="flex items-center space-x-4">
            <Link 
              href="#" 
              className="text-text-secondary hover:text-primary transition-colors"
              title="About"
            >
              <FaInfoCircle />
            </Link>            <Link 
              href="https://github.com" 
              target="_blank"
              className="text-text-secondary hover:text-primary transition-colors"
              title="GitHub"
            >
              <FaGithub />
            </Link>
            {user ? (
              <Link 
                href="/dashboard" 
                className="flex items-center space-x-2 bg-surface/50 hover:bg-surface/80 px-3 py-2 rounded-md transition-colors"
              >
                <FaUser />
                <span>Dashboard</span>
              </Link>
            ) : (
              <Link 
                href="/auth/login" 
                className="flex items-center space-x-2 bg-surface/50 hover:bg-surface/80 px-3 py-2 rounded-md transition-colors"
              >
                <FaUser />
                <span>Login</span>
              </Link>
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
              Describe Your 3D Model
            </h1>
            
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
                      style={{ width: `${(generationStep / 4) * 100}%` }}
                    />
                  </div>
                  <div className="text-sm text-text-secondary mt-2">
                    {generationStep === 1 && 'Creating image from text...'}
                    {generationStep === 2 && 'Generating depth map...'}
                    {generationStep === 3 && 'Converting to 3D model...'}
                    {generationStep === 4 && 'Finalizing your model...'}
                  </div>
                </div>
              )}
            </div>
          </div>
            {/* Right Panel - Output */}
          <div className="glass-panel p-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold">
                {viewMode === 'image' ? 'Generated Image' : '3D Model Preview'}
              </h2>              
              {/* Toggle buttons for view mode */}
              {(imageUrl || modelUrl) && (
                <div className="flex bg-surface/30 rounded-md">
                  <button 
                    className={`toggle-button ${viewMode === 'image' ? 'active' : ''}`}
                    onClick={() => setViewMode('image')}
                  >
                    Image
                  </button>
                  <button 
                    className={`toggle-button ${viewMode === '3d' ? 'active' : ''}`}
                    onClick={() => setViewMode('3d')}
                  >
                    3D Model
                  </button>
                </div>
              )}
            </div>
            
            <div className="aspect-square w-full rounded-lg overflow-hidden">              {/* Image Preview Mode */}
              {viewMode === 'image' && imageUrl ? (
                <div className="image-preview-container w-full h-full">
                  <img 
                    src={imageUrl} 
                    alt={prompt || 'Generated image'} 
                    className="w-full h-full object-cover rounded-lg"
                  />
                  
                  {/* Optional: Show depth map in a small corner */}
                  {depthUrl && (
                    <div className="depth-preview">
                      <img 
                        src={depthUrl} 
                        alt="Depth map" 
                        className="w-full h-full object-cover"
                      />
                    </div>
                  )}
                </div>              ) : viewMode === '3d' ? (
                // In 3D mode, use ModelViewer with whatever URL we have
                <ModelViewer modelUrl={modelUrl || imageUrl || ''} />
              ) : (
                <div className="w-full h-full flex items-center justify-center bg-black/40 rounded-lg">
                  {isGenerating ? (
                    <div className="text-center">
                      <div className="inline-block p-3 rounded-full bg-surface/30 mb-4">
                        <FaMagic className="text-3xl animate-spin text-primary" />
                      </div>
                      <p>Creating your {viewMode === 'image' ? 'image' : '3D model'}...</p>
                    </div>
                  ) : (
                    <div className="text-center p-6">
                      <p className="text-lg mb-2">Your {viewMode === 'image' ? 'image' : '3D model'} will appear here</p>
                      <p className="text-sm text-text-secondary">
                        Enter a prompt and click Generate to start
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
            
            {(imageUrl || modelUrl) && (
              <div className="mt-4 space-y-4">
                {viewMode === '3d' && modelUrl && (
                  <div className="flex justify-between text-sm text-text-secondary">
                    <span>Format: OBJ</span>
                    <span>~10,000 polygons</span>
                  </div>
                )}
                
                <div className="flex space-x-2">
                  {viewMode === 'image' && imageUrl ? (
                    <>
                      <a 
                        href={imageUrl} 
                        download={`3dify-image-${Date.now()}.png`}
                        className="flex-1 gradient-button py-2 text-sm flex items-center justify-center space-x-2"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <FaDownload />
                        <span>Download Image</span>
                      </a>                      <button 
                        className="flex-1 bg-surface/50 hover:bg-surface py-2 text-sm rounded-md transition-colors flex items-center justify-center space-x-2"
                        onClick={() => {
                          setViewMode('3d');
                          // Even without a model URL, we'll attempt to show a 3D representation
                          if (!modelUrl) {
                            toast('Showing basic 3D representation of the image', {
                              icon: 'ℹ️',
                              duration: 3000
                            });
                          }
                        }}
                      >
                        <FaMagic />
                        <span>View 3D</span>
                      </button>
                    </>                  ) : viewMode === '3d' ? (
                    <>
                      {modelUrl ? (
                        <a 
                          href={modelUrl} 
                          download={`3dify-model-${Date.now()}.obj`}
                          className="flex-1 gradient-button py-2 text-sm flex items-center justify-center space-x-2"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <FaDownload />
                          <span>Download Model</span>
                        </a>
                      ) : (
                        <button
                          className="flex-1 gradient-button py-2 text-sm flex items-center justify-center space-x-2 opacity-50"
                          disabled
                        >
                          <FaDownload />
                          <span>Model Unavailable</span>
                        </button>
                      )}
                      <button 
                        className="flex-1 bg-surface/50 hover:bg-surface py-2 text-sm rounded-md transition-colors flex items-center justify-center space-x-2"
                        onClick={() => {
                          setViewMode('image');
                          toast.success('Switched to image view');
                        }}
                      >
                        <FaShare />
                        <span>View Image</span>
                      </button>
                    </>
                  ) : null}
                </div>              </div>
            )}
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
