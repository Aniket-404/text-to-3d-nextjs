'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import useAuth from '@/hooks/useAuth';
import { getFirebaseFirestore } from '@/lib/firebase';
import { collection, query, where, getDocs, orderBy, deleteDoc, doc } from 'firebase/firestore';
import toast from 'react-hot-toast';
import { FaSpinner, FaTrash, FaEye, FaSignOutAlt, FaPlus, FaDownload } from 'react-icons/fa';
import { getRelatedModelFiles } from '@/lib/cloudinaryUtils';

interface Model {
  id: string;
  prompt: string;
  imagePath: string;
  depthPath: string;
  modelPath: string;
  createdAt: any;
}

export default function DashboardPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const { user, signOut } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!user) {
      router.push('/auth/login');
      return;
    }

    const fetchModels = async () => {
      try {
        const db = getFirebaseFirestore();
        if (!db) {
          console.error('Firestore not initialized');
          return;
        }
        
        const q = query(
          collection(db, 'models'),
          where('userId', '==', user.uid),
          orderBy('createdAt', 'desc')
        );
        
        const querySnapshot = await getDocs(q);
        const modelData: Model[] = [];
        
        querySnapshot.forEach((doc) => {
          modelData.push({
            id: doc.id,
            ...doc.data(),
          } as Model);
        });
        
        setModels(modelData);
      } catch (error) {
        console.error('Error fetching models:', error);
        toast.error('Failed to fetch your models');
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, [user, router]);

  const handleDeleteModel = async (id: string) => {
    if (confirm('Are you sure you want to delete this model?')) {
      try {
        const db = getFirebaseFirestore();
        if (!db) {
          console.error('Firestore not initialized');
          return;
        }
        
        await deleteDoc(doc(db, 'models', id));
        setModels(models.filter(model => model.id !== id));
        toast.success('Model deleted successfully');
      } catch (error) {
        console.error('Error deleting model:', error);
        toast.error('Failed to delete model');
      }
    }
  };

  const handleSignOut = async () => {
    try {
      await signOut();
      toast.success('Signed out successfully');
      router.push('/');
    } catch (error) {
      console.error('Error signing out:', error);
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
            <button
              onClick={handleSignOut}
              className="flex items-center space-x-2 bg-surface/50 hover:bg-surface/80 px-3 py-2 rounded-md transition-colors"
            >
              <FaSignOutAlt />
              <span>Sign Out</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-grow container mx-auto px-4 py-10">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold text-gradient">Your 3D Models</h1>
          
          <Link
            href="/"
            className="gradient-button px-4 py-2 flex items-center space-x-2"
          >
            <FaPlus />
            <span>Create New Model</span>
          </Link>
        </div>
        
        {loading ? (
          <div className="flex justify-center items-center h-64">
            <FaSpinner className="animate-spin text-4xl text-primary" />
          </div>
        ) : models.length === 0 ? (
          <div className="glass-panel p-10 text-center">
            <h2 className="text-2xl mb-4">No models yet</h2>
            <p className="text-text-secondary mb-6">
              Create your first 3D model by entering a text prompt
            </p>
            <Link
              href="/"
              className="gradient-button px-6 py-3 inline-flex items-center space-x-2"
            >
              <FaPlus />
              <span>Create Your First Model</span>
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">            {models.map((model) => {
              // Check if we have paths in the standard format
              const { imageUrl } = getRelatedModelFiles(model.imagePath || '');
              const displayImageUrl = imageUrl || model.imagePath;
              
              return (
                <div key={model.id} className="glass-panel overflow-hidden">
                  <div className="relative aspect-square">
                    {displayImageUrl ? (
                      <Image
                        src={displayImageUrl}
                        alt={model.prompt}
                        fill
                        className="object-cover"
                      />
                    ) : (
                      <div className="w-full h-full bg-surface flex items-center justify-center">
                        <span>No image</span>
                      </div>
                    )}
                  </div>
                  
                  <div className="p-4">
                    <h3 className="font-bold mb-2 truncate">{model.prompt}</h3>
                    <p className="text-xs text-gray-400">
                      {model.createdAt ? new Date(model.createdAt.seconds * 1000).toLocaleString() : 'Unknown date'}
                    </p>
                    
                    <div className="flex justify-between mt-4">
                      <Link
                        href={`/models/${model.id}`}
                        className="flex items-center space-x-1 text-primary hover:underline"
                      >
                        <FaEye />
                        <span>View</span>
                      </Link>
                      
                      {model.modelPath && (
                        <a 
                          href={model.modelPath}
                          download
                          className="flex items-center space-x-1 text-green-500 hover:text-green-400"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <FaDownload />
                          <span>OBJ</span>
                        </a>
                      )}
                      
                      <button
                        onClick={() => handleDeleteModel(model.id)}
                        className="flex items-center space-x-1 text-red-500 hover:text-red-400"
                      >
                        <FaTrash />
                        <span>Delete</span>
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
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
