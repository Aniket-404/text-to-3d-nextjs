'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { FaUpload, FaTimes, FaImage } from 'react-icons/fa';

interface ImageUploadProps {
  onImageUpload: (file: File) => void;
  isUploading: boolean;
  uploadedImageUrl?: string;
  onRemoveImage?: () => void;
}

export default function ImageUpload({
  onImageUpload,
  isUploading,
  uploadedImageUrl,
  onRemoveImage
}: ImageUploadProps) {
  const [isDragActive, setIsDragActive] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      onImageUpload(file);
    }
  }, [onImageUpload]);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp']
    },
    maxFiles: 1,
    multiple: false,
    onDragEnter: () => setIsDragActive(true),
    onDragLeave: () => setIsDragActive(false),
    onDropAccepted: () => setIsDragActive(false),
    onDropRejected: () => setIsDragActive(false)
  });

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onImageUpload(file);
    }
    // Reset the input value to allow selecting the same file again
    event.target.value = '';
  };

  if (uploadedImageUrl) {
    return (
      <div className="relative">
        <div className="relative rounded-lg overflow-hidden border-2 border-white/10 bg-surface/30 h-64">
          <img
            src={uploadedImageUrl}
            alt="Uploaded image"
            className={`w-full h-full object-cover transition-opacity ${isUploading ? 'opacity-70' : ''}`}
          />
          {isUploading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
              <div className="text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
                <p className="text-sm text-white">Converting to 3D...</p>
              </div>
            </div>
          )}
          {onRemoveImage && !isUploading && (
            <button
              onClick={onRemoveImage}
              className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-2 hover:bg-red-600 transition-colors"
            >
              <FaTimes />
            </button>
          )}
        </div>
        <p className="text-sm text-text-secondary mt-2 text-center">
          {isUploading ? 'Converting to 3D model...' : 'Image uploaded successfully'}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Drag and Drop Area */}
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all
          ${isDragActive 
            ? 'border-primary bg-primary/10' 
            : 'border-white/20 hover:border-white/30'
          }
          ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}
          bg-surface/20
        `}
      >
        <input {...getInputProps()} disabled={isUploading} />
        
        <div className="space-y-4">
          <div className="flex justify-center">
            {isUploading ? (
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
            ) : (
              <FaImage size={48} className="text-text-secondary" />
            )}
          </div>
          
          <div>
            <p className="text-lg font-medium">
              {isUploading ? 'Uploading...' : 'Drop your image here'}
            </p>
            <p className="text-sm text-text-secondary mt-1">
              {isUploading 
                ? 'Please wait while we upload your image...' 
                : 'Or click to browse and select an image file'
              }
            </p>
          </div>
          
          <div className="text-xs text-text-secondary">
            Supports: JPEG, PNG, GIF, BMP, WebP
          </div>
        </div>
      </div>

      {/* Alternative Upload Button */}
      <div className="flex justify-center">
        <label className={`
          inline-flex items-center px-4 py-2 border border-white/20 rounded-md shadow-sm text-sm font-medium bg-surface/50 hover:bg-surface/70 cursor-pointer transition-colors
          ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}>
          <FaUpload className="mr-2" />
          Choose Image File
          <input
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            disabled={isUploading}
            className="sr-only"
          />
        </label>
      </div>
    </div>
  );
}