import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  // This is a simplified middleware for demonstration
  // In a real app, you would need to verify the Firebase token on the server
  
  const path = request.nextUrl.pathname;
  
  // Protect dashboard and model routes
  if (path.startsWith('/dashboard') || path.startsWith('/models/')) {
    // In a production app, you would check for a valid session cookie
    // For now, we'll rely on client-side auth in the components
    
    // This is just a placeholder for future enhancement
    // const authToken = request.cookies.get('authToken')?.value;
    // if (!authToken) {
    //   return NextResponse.redirect(new URL('/auth/login', request.url));
    // }
  }
  
  return NextResponse.next();
}

export const config = {
  matcher: ['/dashboard/:path*', '/models/:path*'],
};
