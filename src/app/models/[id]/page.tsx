'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import Image from 'next/image';
import ModelViewer from '@/components/ModelViewer';
import useAuth from '@/hooks/useAuth';
import { getFirebaseFirestore } from '@/lib/firebase';
import { doc, getDoc } from 'firebase/firestore';
import { FaArrowLeft, FaSpinner, FaImage, FaCube, FaDownload } from 'react-icons/fa';
import { getRelatedModelFiles } from '@/lib/cloudinaryUtils';
import toast from 'react-hot-toast';

export default function ModelViewPage({ params }: { params: { id: string } }) {
  const { id } = params;
  const router = useRouter();
  const { user } = useAuth();
  const [model, setModel] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'image' | '3d'>('image');
  const [hasError, setHasError] = useState(false);
  useEffect(() => {
    const fetchModel = async () => {
      if (!user) {
        router.push('/auth/login');
        return;
      }

      try {
        const db = getFirebaseFirestore();
        if (!db) throw new Error('Firestore not initialized');
        
        const docRef = doc(db, 'models', id);
        const docSnap = await getDoc(docRef);
        
        if (!docSnap.exists()) {
          setHasError(true);
          toast.error('Model not found');
          return;
        }
        
        const docData = docSnap.data();
        const modelData = {
          id: docSnap.id,
          ...docData,
          imagePath: docData.imagePath || '',
          depthPath: docData.depthPath || '',
          modelPath: docData.modelPath || '',
          prompt: docData.prompt || 'Untitled Model',
          createdAt: docData.createdAt?.toDate() || new Date()
        };
        
        // Ensure we have all the related model files
        if (modelData.imagePath) {
          const relatedFiles = getRelatedModelFiles(modelData.imagePath);
          if (relatedFiles.imageUrl) modelData.imagePath = relatedFiles.imageUrl;
          if (relatedFiles.depthUrl) modelData.depthPath = relatedFiles.depthUrl;
          if (relatedFiles.modelUrl) modelData.modelPath = relatedFiles.modelUrl;
        }
        
        setModel(modelData);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching model:', error);
        setHasError(true);
        toast.error('Failed to load model');
        setLoading(false);
      }
    };

    fetchModel();
  }, [id, user, router]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen">
        <FaSpinner className="animate-spin text-4xl text-blue-500 mb-4" />
        <p>Loading model...</p>
      </div>
    );
  }

  if (hasError || !model) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen">
        <div className="glass-panel p-8 max-w-md text-center">
          <h1 className="text-2xl font-bold mb-4">Model Not Found</h1>
          <p className="mb-6">The model you're looking for doesn't exist or you don't have permission to view it.</p>
          <Link
            href="/dashboard"
            className="gradient-button px-4 py-2 flex items-center space-x-2 justify-center"
          >
            <FaArrowLeft />
            <span>Back to Dashboard</span>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-6">
        <Link
          href="/dashboard"
          className="inline-flex items-center text-primary hover:underline"
        >
          <FaArrowLeft className="mr-2" />
          Back to Dashboard
        </Link>
      </div>
      
      <div className="glass-panel p-6">
        <h1 className="text-2xl font-bold mb-2">{model.prompt}</h1>
        <p className="text-sm text-gray-400 mb-6">
          Created: {model.createdAt.toLocaleString()}
        </p>
        
        <div className="mb-6">
          <div className="flex space-x-2 mb-4">
            <button
              onClick={() => setViewMode('image')}
              className={`px-4 py-2 rounded-md flex items-center space-x-2 ${
                viewMode === 'image' 
                  ? 'bg-primary text-white' 
                  : 'bg-surface hover:bg-surface-hover'
              }`}
            >
              <FaImage />
              <span>Image</span>
            </button>
            
            <button
              onClick={() => setViewMode('3d')}
              className={`px-4 py-2 rounded-md flex items-center space-x-2 ${
                viewMode === '3d' 
                  ? 'bg-primary text-white' 
                  : 'bg-surface hover:bg-surface-hover'
              }`}
            >
              <FaCube />
              <span>3D Model</span>
            </button>
            
            {model.modelPath && (
              <a
                href={model.modelPath}
                download
                className="px-4 py-2 rounded-md bg-green-600 text-white flex items-center space-x-2 hover:bg-green-700"
                target="_blank"
                rel="noopener noreferrer"
              >
                <FaDownload />
                <span>Download OBJ</span>
              </a>
            )}
          </div>
          
          {viewMode === 'image' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {model.imagePath && (
                <div className="glass-panel p-4">
                  <h3 className="text-lg font-medium mb-2">Generated Image</h3>
                  <div className="relative aspect-square">
                    <Image 
                      src={model.imagePath} 
                      alt={model.prompt}
                      fill
                      className="object-contain"
                    />
                  </div>
                </div>
              )}
              
              {model.depthPath && (
                <div className="glass-panel p-4">
                  <h3 className="text-lg font-medium mb-2">Depth Map</h3>
                  <div className="relative aspect-square">
                    <Image 
                      src={model.depthPath} 
                      alt="Depth map"
                      fill
                      className="object-contain"
                    />
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="aspect-square bg-black rounded-lg overflow-hidden">
              {model.modelPath ? (
                <ModelViewer modelUrl={model.modelPath} initialAutoRotate={true} />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <p>3D model not available</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
