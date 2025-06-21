import { v2 as cloudinary } from 'cloudinary';

cloudinary.config({
  cloud_name: process.env.NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

export const uploadToCloudinary = async (file: File, customFolder?: string, customPublicId?: string) => {
  // Create a FormData object and append the file
  const formData = new FormData();
  formData.append('file', file);
  formData.append('upload_preset', 'text-to-3d');
  
  // Add folder if specified
  if (customFolder) {
    formData.append('folder', customFolder);
  } else {
    formData.append('folder', 'text-to-3d-web/models');
  }
  
  // Add public_id if specified
  if (customPublicId) {
    formData.append('public_id', customPublicId);
  }

  // Make the upload request
  const response = await fetch(
    `https://api.cloudinary.com/v1_1/${process.env.NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME}/upload`,
    {
      method: 'POST',
      body: formData,
    }
  );

  if (!response.ok) {
    throw new Error('Failed to upload file to Cloudinary');
  }

  const data = await response.json();
  return data.secure_url;
};

export default cloudinary;
