'use client';

// Utility functions to standardize Cloudinary URL handling for 3D model generation

/**
 * Extracts information from a Cloudinary URL
 * @param url - The Cloudinary URL to parse
 * @returns An object containing file type, prompt, and timestamp
 */
export function parseCloudinaryUrl(url: string) {
  try {
    const urlParts = url.split('/');
    const filename = urlParts[urlParts.length - 1].split('.')[0]; // Get filename without extension
    
    // Try to extract type, prompt and timestamp
    // Format is typically: type_prompt_timestamp
    // For example: image_mountain_landscape_1234567890
    const firstUnderscore = filename.indexOf('_');
    if (firstUnderscore === -1) return { type: 'unknown', prompt: 'unknown', timestamp: 0 };
    
    const type = filename.substring(0, firstUnderscore);
    
    // Find the last underscore to extract timestamp
    const lastUnderscore = filename.lastIndexOf('_');
    if (lastUnderscore === firstUnderscore) return { type, prompt: 'unknown', timestamp: 0 };
    
    const prompt = filename.substring(firstUnderscore + 1, lastUnderscore);
    const timestamp = parseInt(filename.substring(lastUnderscore + 1), 10) || 0;
    
    return { type, prompt, timestamp };
  } catch (error) {
    console.error('Error parsing Cloudinary URL:', error);
    return { type: 'unknown', prompt: 'unknown', timestamp: 0 };
  }
}

/**
 * Creates a standardized public ID for Cloudinary uploads
 * @param type - The type of file (image, depth, model)
 * @param prompt - The text prompt used to generate the model
 * @returns A formatted public ID string
 */
export function createCloudinaryPublicId(type: string, prompt: string): string {
  const timestamp = Math.floor(Date.now() / 1000);
  const sanitizedPrompt = prompt
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .replace(/\s+/g, '_')
    .substring(0, 30)
    .trim();
  
  return `${type}_${sanitizedPrompt}_${timestamp}`;
}

/**
 * Retrieves all related files for a 3D model set
 * @param url - Any Cloudinary URL from the model set
 * @returns An object with URLs for the image, depth map, and 3D model
 */
export function getRelatedModelFiles(url: string) {
  try {
    const { prompt, timestamp } = parseCloudinaryUrl(url);
    if (prompt === 'unknown' || timestamp === 0) {
      return {
        imageUrl: null,
        depthUrl: null,
        modelUrl: null
      };
    }
    
    // Extract the base URL (everything before the public ID)
    const baseUrlParts = url.split('/');
    baseUrlParts.pop(); // Remove the filename
    const baseUrl = baseUrlParts.join('/');
    
    // Create URLs for each file type
    return {
      imageUrl: `${baseUrl}/image_${prompt}_${timestamp}.png`,
      depthUrl: `${baseUrl}/depth_${prompt}_${timestamp}.png`,
      modelUrl: `${baseUrl}/model_${prompt}_${timestamp}.obj`
    };
  } catch (error) {
    console.error('Error getting related model files:', error);
    return {
      imageUrl: null,
      depthUrl: null,
      modelUrl: null
    };
  }
}

/**
 * Generates a sanitized Cloudinary path for the given prompt
 * @param prompt - The text prompt to sanitize
 * @returns Object containing paths for image, depth map, and 3D model
 */
export function generateCloudinaryPaths(prompt: string) {
  const sanitizedPrompt = prompt
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .replace(/\s+/g, '_')
    .substring(0, 30)
    .trim();
    
  const timestamp = Math.floor(Date.now() / 1000);
  const folder = 'text-to-3d-web/models';
  
  return {
    imagePublicId: `image_${sanitizedPrompt}_${timestamp}`,
    depthPublicId: `depth_${sanitizedPrompt}_${timestamp}`,
    modelPublicId: `model_${sanitizedPrompt}_${timestamp}`,
    folder
  };
}
