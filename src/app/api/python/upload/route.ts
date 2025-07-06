import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }

    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      return NextResponse.json(
        { error: 'Invalid file type. Please upload a valid image file (JPEG, PNG, GIF, BMP, WebP)' },
        { status: 400 }
      );
    }

    // Validate file size (10MB limit)
    const maxSize = 10 * 1024 * 1024; // 10MB in bytes
    if (file.size > maxSize) {
      return NextResponse.json(
        { error: 'File size too large. Maximum size is 10MB.' },
        { status: 400 }
      );
    }

    // Create FormData for the Python API
    const pythonFormData = new FormData();
    pythonFormData.append('file', file);

    // Forward the request to the Python API
    const pythonApiUrl = process.env.PYTHON_API_URL || 'http://localhost:5000';
    const response = await fetch(`${pythonApiUrl}/upload`, {
      method: 'POST',
      body: pythonFormData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `Python API error: ${response.status}`);
    }

    const data = await response.json();
    
    return NextResponse.json(data);
  } catch (error: any) {
    console.error('Upload API Error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to process upload' },
      { status: 500 }
    );
  }
}
