import { NextRequest, NextResponse } from 'next/server';

const PYTHON_API_BASE_URL = process.env.PYTHON_API_BASE_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { prompt, image_url, depth_model = 'intel', steps = 3000, resolution = 512 } = body;

    if (!prompt) {
      return NextResponse.json(
        { error: 'Prompt is required' },
        { status: 400 }
      );
    }

    console.log('🧠 Starting NeRF generation:', {
      prompt: prompt.substring(0, 50) + '...',
      depth_model,
      steps,
      resolution,
      has_image: !!image_url
    });

    // Forward request to Python API
    const pythonResponse = await fetch(`${PYTHON_API_BASE_URL}/nerf/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt,
        image_url,
        depth_model,
        steps,
        resolution,
        include_weights: true,
        include_config: true,
        include_mesh: true,
        include_viewer: true
      }),
    });

    if (!pythonResponse.ok) {
      const errorText = await pythonResponse.text();
      console.error('❌ Python API error:', errorText);
      
      if (pythonResponse.status === 503) {
        return NextResponse.json(
          { error: 'Python API is not running. Please start the backend server.' },
          { status: 503 }
        );
      }
      
      return NextResponse.json(
        { error: errorText || 'NeRF generation failed' },
        { status: pythonResponse.status }
      );
    }

    const result = await pythonResponse.json();
    console.log('✅ NeRF generation initiated:', result.job_id);

    return NextResponse.json(result);
  } catch (error) {
    console.error('❌ NeRF generation error:', error);
    
    if (error instanceof Error) {
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        return NextResponse.json(
          { error: 'Unable to connect to Python API. Please ensure the backend server is running.' },
          { status: 503 }
        );
      }
      
      return NextResponse.json(
        { error: error.message },
        { status: 500 }
      );
    }
    
    return NextResponse.json(
      { error: 'An unexpected error occurred during NeRF generation' },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json(
    { error: 'Method not allowed. Use POST to generate NeRF models.' },
    { status: 405 }
  );
}
