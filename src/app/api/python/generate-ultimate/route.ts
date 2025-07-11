import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Validate required fields
    if (!body.prompt) {
      return NextResponse.json(
        { error: 'Prompt is required' },
        { status: 400 }
      );
    }

    // Validate parameters
    const numViews = body.num_views || 8;
    const nerfSteps = body.nerf_steps || 5000;
    const resolution = body.resolution || 1024;

    if (numViews < 4 || numViews > 12) {
      return NextResponse.json(
        { error: 'Number of views must be between 4 and 12' },
        { status: 400 }
      );
    }

    if (nerfSteps < 1000 || nerfSteps > 10000) {
      return NextResponse.json(
        { error: 'NeRF steps must be between 1000 and 10000' },
        { status: 400 }
      );
    }

    if (![512, 1024].includes(resolution)) {
      return NextResponse.json(
        { error: 'Resolution must be 512 or 1024 for ultimate quality' },
        { status: 400 }
      );
    }

    // Forward to Python backend
    const pythonApiUrl = process.env.PYTHON_API_URL || 'http://localhost:5000';
    
    const response = await fetch(`${pythonApiUrl}/ultimate/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: body.prompt,
        num_views: numViews,
        nerf_steps: nerfSteps,
        resolution: resolution,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Ultimate generation failed');
    }

    const data = await response.json();
    
    return NextResponse.json(data);
    
  } catch (error: any) {
    console.error('Ultimate generation API error:', error);
    
    // Handle specific error cases
    if (error.message?.includes('ECONNREFUSED') || error.message?.includes('fetch')) {
      return NextResponse.json(
        { 
          error: 'Python API is not running. Please start the backend server.',
          details: 'Run: cd python-api && python app.py'
        },
        { status: 503 }
      );
    }
    
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}
