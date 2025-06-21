import { NextResponse } from 'next/server';

export const maxDuration = 300; // Set max duration to 5 minutes

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { prompt } = body;
    
    if (!prompt) {
      return NextResponse.json(
        { error: 'Prompt is required' }, 
        { status: 400 }
      );
    }
    
    // Get the API URL from environment variables
    const apiUrl = process.env.NEXT_PUBLIC_API_URL;
    if (!apiUrl) {
      console.error('API URL is not configured');
      return NextResponse.json(
        { error: 'API URL is not configured' },
        { status: 500 }
      );
    }
    
    console.log(`Calling Python API at: ${apiUrl}/generate`);
    
    // First check if the API is running
    try {
      const healthCheck = await fetch(`${apiUrl}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!healthCheck.ok) {
        console.error('Python API is not running or not accessible');
        return NextResponse.json(
          { error: 'Python API is not running. Please make sure the Python server is started.' },
          { status: 503 }
        );
      }
      
      const healthData = await healthCheck.json();
      console.log('API Health Check:', healthData);
    } catch (error) {
      console.error('Failed to connect to Python API:', error);
      return NextResponse.json(
        { error: 'Failed to connect to Python API. Please make sure the server is running on port 5000.' },
        { status: 503 }
      );
    }
    
    try {
      const response = await fetch(`${apiUrl}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });
    
      if (!response.ok) {
        console.error('API Error Status:', response.status);
        const errorText = await response.text();
        let errorMessage = 'Failed to generate content';
        
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.error || errorMessage;
        } catch (e) {
          // If the response isn't valid JSON, use the raw text
          errorMessage = errorText || errorMessage;
        }
        
        console.error('API Error Message:', errorMessage);
        
        return NextResponse.json(
          { error: errorMessage },
          { status: response.status }
        );
      }
      
      const data = await response.json();
      return NextResponse.json(data);
      
    } catch (error: any) {
      console.error('Fetch error:', error);
      return NextResponse.json(
        { 
          error: error.name === 'AbortError' 
            ? 'Image generation is taking longer than usual. Please try again.' 
            : 'Failed to communicate with Python API'
        },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error('Request error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
