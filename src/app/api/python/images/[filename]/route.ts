import { NextResponse } from 'next/server';
import { NextRequest } from 'next/server';

export async function GET(
  request: NextRequest,
  { params }: { params: { filename: string } }
) {
  try {
    const filename = params.filename;
    
    if (!filename) {
      return NextResponse.json(
        { error: 'Filename is required' }, 
        { status: 400 }
      );
    }
    
    // Call the Python API to retrieve the image
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/images/${filename}`);
    
    if (!response.ok) {
      return NextResponse.json(
        { error: 'Failed to retrieve image' }, 
        { status: response.status }
      );
    }
    
    // Get the content type from the Python API response
    const contentType = response.headers.get('content-type') || 'application/octet-stream';
    
    // Forward the response from the Python API
    const data = await response.arrayBuffer();
    
    return new NextResponse(data, {
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=31536000, immutable',
      },
    });
  } catch (error: any) {
    console.error('Error retrieving image:', error);
    return NextResponse.json(
      { error: error.message || 'Internal server error' }, 
      { status: 500 }
    );
  }
}
