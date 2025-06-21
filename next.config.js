/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: ['res.cloudinary.com'],
  },
  transpilePackages: ['firebase'],
  webpack: (config, { isServer }) => {
    // Resolve issues with private class fields in undici
    if (!isServer) {
      config.resolve.alias = {
        ...config.resolve.alias,
        // Avoid issues with undici's use of private class fields
        undici: false,
      };
    }
    return config;
  },
  // This is needed for the Python API service
  async rewrites() {
    return [
      {
        source: '/api/python/:path*',
        destination: 
          process.env.NODE_ENV === 'production'
            ? `${process.env.NEXT_PUBLIC_API_URL}/:path*` // Use the deployed API URL
            : 'http://localhost:5000/:path*', // Use local Python API in development
      },
    ];
  },
  // Use strict mode for better development experience
  reactStrictMode: true,
  // Use swcMinify for better performance
  swcMinify: true,
  // Handle third-party script attributes properly
  compiler: {
    // This will help with styled components
    styledComponents: true,
    removeConsole: process.env.NODE_ENV === 'production',
  },
  // Disable HTML validation errors from extensions like Grammarly
  onDemandEntries: {
    // Keep pages in memory for longer during development
    maxInactiveAge: 25 * 1000,
    pagesBufferLength: 2,
  },
};

module.exports = nextConfig;
