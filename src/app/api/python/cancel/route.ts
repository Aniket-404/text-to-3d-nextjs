import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { job_id } = body;
    
    if (!job_id) {
      return NextResponse.json(
        { error: 'Job ID is required' }, 
        { status: 400 }
      );
    }
    
    // Get the API URL from environment variables
    const pythonApiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';
    
    console.log(`Cancelling job: ${job_id}`);
    
    try {
      const response = await fetch(`${pythonApiUrl}/cancel`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ job_id }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Failed to cancel job: ${response.status}`);
      }
      
      const data = await response.json();
      return NextResponse.json(data);
      
    } catch (error: any) {
      console.error('Cancel request error:', error);
      return NextResponse.json(
        { error: error.message || 'Failed to cancel job' },
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
