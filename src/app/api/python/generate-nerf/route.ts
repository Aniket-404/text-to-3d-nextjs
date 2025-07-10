import { NextRequest, NextResponse } from 'next/server';

const PYTHON_API_BASE_URL = process.env.PYTHON_API_BASE_URL || 'http://localhost:5000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { prompt, image_url, depth_model = 'intel', steps = 3000, resolution = 512 } = body;

    if (!prompt || prompt.trim().length === 0) {
      return NextResponse.json(
        { error: 'A valid prompt is required for NeRF generation' },
        { status: 400 }
      );
    }

    // Validate parameters
    if (steps < 500 || steps > 10000) {
      return NextResponse.json(
        { error: 'Steps must be between 500 and 10000' },
        { status: 400 }
      );
    }

    if (![256, 512, 1024].includes(resolution)) {
      return NextResponse.json(
        { error: 'Resolution must be 256, 512, or 1024' },
        { status: 400 }
      );
    }

    console.log('🧠 Starting production NeRF generation:', {
      prompt: prompt.substring(0, 50) + '...',
      depth_model,
      steps,
      resolution,
      has_image: !!image_url
    });

    // Forward request to Python API with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minute timeout

    try {
      const pythonResponse = await fetch(`${PYTHON_API_BASE_URL}/nerf/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: prompt.trim(),
          image_url,
          depth_model,
          steps,
          resolution,
          include_weights: true,
          include_config: true,
          include_mesh: true,
          include_viewer: true,
          production: true
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!pythonResponse.ok) {
        const errorText = await pythonResponse.text();
        console.error('❌ Python API error:', errorText);
        
        if (pythonResponse.status === 503) {
          return NextResponse.json(
            { error: 'NeRF training service is unavailable. Please try again later.' },
            { status: 503 }
          );
        }
        
        if (pythonResponse.status === 429) {
          return NextResponse.json(
            { error: 'NeRF training queue is full. Please wait and try again.' },
            { status: 429 }
          );
        }
        
        return NextResponse.json(
          { error: errorText || 'NeRF generation failed' },
          { status: pythonResponse.status }
        );
      }

      const result = await pythonResponse.json();
      console.log('✅ Production NeRF generation initiated:', result.job_id);

      return NextResponse.json({
        ...result,
        estimated_completion: new Date(Date.now() + (steps * 100)).toISOString() // Rough estimate
      });

    } catch (fetchError) {
      clearTimeout(timeoutId);
      if (controller.signal.aborted) {
        return NextResponse.json(
          { error: 'NeRF generation timed out. Please try with fewer steps or lower resolution.' },
          { status: 408 }
        );
      }
      throw fetchError;
    }

  } catch (error) {
    console.error('❌ NeRF generation error:', error);
    
    if (error instanceof Error) {
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        return NextResponse.json(
          { error: 'Unable to connect to NeRF training service. Please ensure the backend server is running.' },
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
