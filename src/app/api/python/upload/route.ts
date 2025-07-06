import { NextResponse } from 'next/server';
import http from 'http';
import https from 'https';
import { URL } from 'url';

export const maxDuration = 600; // Set max duration to 10 minutes for depth processing

// Helper function to make HTTP requests with custom timeout
function makeRequest(url: string, options: any, data?: string): Promise<{ statusCode?: number; data: string }> {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(url);
    const isHttps = parsedUrl.protocol === 'https:';
    const client = isHttps ? https : http;
    
    const requestOptions = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port || (isHttps ? 443 : 80),
      path: parsedUrl.pathname + parsedUrl.search,
      method: options.method || 'GET',
      headers: options.headers || {},
      timeout: 600000, // 10 minutes timeout
    };
    
    const req = client.request(requestOptions, (res) => {
      let responseData = '';
      
      res.on('data', (chunk) => {
        responseData += chunk;
      });
      
      res.on('end', () => {
        resolve({
          statusCode: res.statusCode,
          data: responseData
        });
      });
    });
    
    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Request timeout after 10 minutes'));
    });
    
    req.on('error', (error) => {
      reject(error);
    });
    
    if (data) {
      req.write(data);
    }
    
    req.end();
  });
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const imageFile = formData.get('image') as File;
    
    if (!imageFile) {
      return NextResponse.json(
        { error: 'No image file provided' }, 
        { status: 400 }
      );
    }
    
    // Convert file to base64
    const bytes = await imageFile.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const base64Image = buffer.toString('base64');
    
    // Use localhost for local development
    const apiUrl = process.env.NODE_ENV === 'production' 
      ? process.env.NEXT_PUBLIC_API_URL 
      : 'http://localhost:5000';
    
    if (!apiUrl) {
      console.error('API URL is not configured');
      return NextResponse.json(
        { error: 'API URL is not configured' },
        { status: 500 }
      );
    }
    
    console.log(`Uploading image to Python API at: ${apiUrl}/upload`);
    
    // First check if the API is running
    try {
      const healthResponse = await makeRequest(`${apiUrl}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!healthResponse.statusCode || healthResponse.statusCode >= 400) {
        console.error('Python API is not running or not accessible');
        return NextResponse.json(
          { error: 'Python API is not running. Please make sure the Python server is started.' },
          { status: 503 }
        );
      }
    } catch (error) {
      console.error('Failed to connect to Python API:', error);
      return NextResponse.json(
        { error: 'Failed to connect to Python API. Please make sure the server is running on port 5000.' },
        { status: 503 }
      );
    }
    
    try {
      const response = await makeRequest(`${apiUrl}/upload`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      }, JSON.stringify({ 
        image: base64Image,
        filename: imageFile.name,
        mimetype: imageFile.type
      }));
    
      if (!response.statusCode || response.statusCode >= 400) {
        console.error('API Error Status:', response.statusCode);
        let errorMessage = 'Failed to process uploaded image';
        
        try {
          const errorData = JSON.parse(response.data);
          errorMessage = errorData.error || errorMessage;
        } catch (e) {
          errorMessage = response.data || errorMessage;
        }
        
        console.error('API Error Message:', errorMessage);
        
        return NextResponse.json(
          { error: errorMessage },
          { status: response.statusCode || 500 }
        );
      }
      
      const data = JSON.parse(response.data);
      return NextResponse.json(data);
      
    } catch (error: any) {
      console.error('Request error:', error);
      const isTimeout = error.message?.includes('timeout') || error.message?.includes('Request timeout');
      return NextResponse.json(
        { 
          error: isTimeout
            ? 'Image processing is taking longer than usual (>10 minutes). The process may still be running on the server.' 
            : 'Failed to communicate with Python API: ' + error.message
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
