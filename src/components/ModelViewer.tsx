'use client';

import React, { useRef, useEffect, useState, Suspense, Component, ErrorInfo, ReactNode } from 'react';
import { Canvas, useLoader, useThree, useFrame } from '@react-three/fiber';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { OrbitControls, Environment, ContactShadows, Html } from '@react-three/drei';
import * as THREE from 'three';
import { FaSpinner, FaCompress, FaExpand, FaSyncAlt } from 'react-icons/fa';
import { getRelatedModelFiles } from '@/lib/cloudinaryUtils';

// Error boundary component for catching React rendering errors
class ErrorBoundary extends Component<{ 
  fallback: ReactNode, 
  children: ReactNode 
}, { hasError: boolean }> {
  constructor(props: { fallback: ReactNode, children: ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(_: Error) {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Three.js Error Boundary caught error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }

    return this.props.children;
  }
}

// Fallback model component to show when loading fails
function FallbackModel() {
  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="red" />
      <Html position={[0, 0, 0]}>
        <div style={{ color: 'white', background: 'rgba(0,0,0,0.7)', padding: '10px', borderRadius: '4px' }}>
          Error loading model
        </div>
      </Html>
    </mesh>
  );
}

function Model({ url, autoRotate }: { url: string, autoRotate: boolean }) {
  const [error, setError] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const { camera } = useThree();
  const groupRef = useRef<THREE.Group>(new THREE.Group());
  
  // Create a manual loader instead of using useLoader to have more control
  useEffect(() => {
    let isMounted = true;
    
    const loadModel = async () => {
      try {
        // Use a manual loader for better control over the process
        const loader = new OBJLoader();
        
        // Progress handler that doesn't trigger errors
        loader.loadAsync(url)
          .then((obj) => {
            if (!isMounted) return;
            
            // Position camera to see the model properly
            camera.position.z = 5;
            
            try {
              // Get the bounding box of the model
              const box = new THREE.Box3().setFromObject(obj);
              const center = new THREE.Vector3();
              box.getCenter(center);
              
              // Center the model
              obj.position.sub(center);
              
              // Scale the model to fit in view
              const size = new THREE.Vector3();
              box.getSize(size);
              const maxDim = Math.max(size.x, size.y, size.z);
              if (maxDim > 0) {
                const scale = 2 / maxDim;
                obj.scale.multiplyScalar(scale);
              }
              
              // Add the model to our group
              groupRef.current.add(obj);
              setLoading(false);
            } catch (setupErr) {
              console.error('Error setting up model:', setupErr);
              setError(true);
              setLoading(false);
            }
          })
          .catch((loadErr) => {
            if (!isMounted) return;
            console.error('Error loading model:', loadErr);
            setError(true);
            setLoading(false);
          });
      } catch (err) {
        if (!isMounted) return;
        console.error('Model loading error:', err);
        setError(true);
        setLoading(false);
      }
    };
    
    loadModel();
    
    return () => {
      isMounted = false;
    };
  }, [url, camera]);
  
  // Rotate the model
  useFrame(() => {
    if (autoRotate && !error && groupRef.current) {
      groupRef.current.rotation.y += 0.005;
    }
  });
  
  if (error) {
    return <FallbackModel />;
  }
  
  if (loading) {
    return (
      <mesh>
        <sphereGeometry args={[0.5, 16, 16]} />
        <meshStandardMaterial color="blue" wireframe />
      </mesh>
    );
  }
    return <primitive object={groupRef.current} />;
}

function SceneContent({ modelUrl, autoRotate }: { modelUrl: string, autoRotate: boolean }) {
  const [modelError, setModelError] = useState<boolean>(false);
  
  // Use this effect to log the model URL for debugging
  useEffect(() => {
    console.log("Loading model from URL:", modelUrl);
    
    // Pre-validate the URL
    const validateUrl = async () => {
      try {
        // Try to fetch the headers only to check if the URL is valid
        const response = await fetch(modelUrl, { method: 'HEAD' });
        if (!response.ok) {
          console.error("Model URL returned status:", response.status);
          setModelError(true);
        }
      } catch (err) {
        console.error("Error validating model URL:", err);
        setModelError(true);
      }
    };
    
    validateUrl();
  }, [modelUrl]);
  
  return (
    <>
      <ambientLight intensity={0.8} />
      <spotLight position={[5, 10, 5]} angle={0.3} penumbra={1} intensity={1} castShadow />
      <pointLight position={[-5, -5, -5]} intensity={0.5} />
      
      <Suspense fallback={
        <mesh>
          <sphereGeometry args={[1, 16, 16]} />
          <meshStandardMaterial color="#444" wireframe />
          <Html position={[0, 0, 0]}>
            <div style={{ color: 'white', background: 'rgba(0,0,0,0.7)', padding: '10px', borderRadius: '4px', whiteSpace: 'nowrap' }}>
              Loading model...
            </div>
          </Html>
        </mesh>
      }>
        {modelError ? (
          <FallbackModel />
        ) : (
          <Model url={modelUrl} autoRotate={autoRotate} />
        )}
        <Environment preset="sunset" />
        <ContactShadows 
          opacity={0.4} 
          scale={10} 
          blur={2} 
          far={10} 
          resolution={256} 
          color="#000000" 
        />
      </Suspense>
      <OrbitControls 
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        autoRotate={autoRotate}
        autoRotateSpeed={1}
      />
    </>
  );
}

export default function ModelViewer({ 
  modelUrl, 
  url,
  initialAutoRotate = true 
}: { 
  modelUrl?: string; 
  url?: string;
  initialAutoRotate?: boolean;
}) {
  const [loading, setLoading] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [autoRotate, setAutoRotate] = useState(initialAutoRotate);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Use either modelUrl or url prop
  const modelPath = modelUrl || url;

  useEffect(() => {
    // Set a timeout to hide the loading spinner if it takes too long
    timerRef.current = setTimeout(() => {
      setLoading(false);
    }, 10000);
    
    // Listen for fullscreen change
    const handleFullscreenChange = () => {
      setFullscreen(document.fullscreenElement !== null);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);
  
  const toggleFullscreen = async () => {
    try {
      if (!document.fullscreenElement && containerRef.current) {
        await containerRef.current.requestFullscreen();
      } else if (document.fullscreenElement) {
        await document.exitFullscreen();
      }
    } catch (err) {
      console.error('Error toggling fullscreen:', err);
    }
  };

  const toggleAutoRotate = () => {
    setAutoRotate(!autoRotate);
  };
  
  const handleModelLoad = () => {
    setLoading(false);
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
  };
    return (
    <div ref={containerRef} className="relative w-full h-full min-h-[300px] model-viewer-container">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/20 backdrop-blur-sm z-10">
          <div className="flex flex-col items-center">
            <FaSpinner className="animate-spin text-4xl text-primary mb-4" />
            <p className="text-text-primary">Loading 3D model...</p>
          </div>
        </div>
      )}
      
      <div className="absolute top-2 right-2 z-20 flex space-x-2">
        <button 
          onClick={toggleAutoRotate}
          className="bg-black/40 hover:bg-black/60 text-text-primary p-2 rounded-full transition-colors hover:shadow-glow"
          title={autoRotate ? "Stop rotation" : "Start rotation"}
        >
          <FaSyncAlt className={autoRotate ? "animate-spin" : ""} />
        </button>
        <button 
          onClick={toggleFullscreen}
          className="bg-black/40 hover:bg-black/60 text-text-primary p-2 rounded-full transition-colors hover:shadow-glow"
          title={fullscreen ? "Exit fullscreen" : "View fullscreen"}
        >
          {fullscreen ? <FaCompress /> : <FaExpand />}
        </button>
      </div>
      
      <Canvas
        shadows
        gl={{ antialias: true }}
        camera={{ position: [0, 0, 5], fov: 50 }}
        onCreated={({ gl }) => {
          gl.toneMapping = THREE.ACESFilmicToneMapping;
          gl.toneMappingExposure = 1.2;
          handleModelLoad();
        }}
        className="w-full h-full"
        onError={(error) => {
          console.error("Canvas error:", error);
          handleModelLoad(); // Hide loading spinner on error
        }}
      >
        <ErrorBoundary fallback={<FallbackModel />}>
          <SceneContent modelUrl={modelPath || ''} autoRotate={autoRotate} />
        </ErrorBoundary>
      </Canvas>
    </div>
  );
}
