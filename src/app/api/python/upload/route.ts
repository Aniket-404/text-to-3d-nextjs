import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const depthModel = formData.get('depth_model') as string;
    
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
    if (depthModel) {
      pythonFormData.append('depth_model', depthModel);
    }

    let jobId: string | null = null;

    // Forward the request to the Python API
    const pythonApiUrl = process.env.PYTHON_API_URL || 'http://localhost:5000';
    
    try {
      const response = await fetch(`${pythonApiUrl}/upload`, {
        method: 'POST',
        body: pythonFormData,
        signal: request.signal, // Pass through the abort signal
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Python API error: ${response.status}`);
      }

      const data = await response.json();
      jobId = data.job_id; // Store the job_id for potential cancellation
      
      return NextResponse.json(data);
    } catch (error: any) {
      console.error('Upload API Error:', error);
      
      // If the request was aborted, try to cancel the backend job
      if (error.name === 'AbortError') {
        if (jobId) {
          // Try to cancel the backend job
          try {
            await fetch(`${pythonApiUrl}/cancel`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ job_id: jobId }),
              signal: AbortSignal.timeout(5000), // 5 second timeout for cancellation
            });
            console.log(`Cancelled backend job: ${jobId}`);
          } catch (cancelError) {
            console.error('Failed to cancel backend job:', cancelError);
          }
        }
        
        return NextResponse.json(
          { error: 'Upload was cancelled', cancelled: true },
          { status: 499 }
        );
      }
      
      return NextResponse.json(
        { error: error.message || 'Failed to process upload' },
        { status: 500 }
      );
    }
  } catch (outerError: any) {
    console.error('Outer Upload API Error:', outerError);
    return NextResponse.json(
      { error: outerError.message || 'Failed to process upload' },
      { status: 500 }
    );
  }
}
