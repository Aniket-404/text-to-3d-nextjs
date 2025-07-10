/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: ['res.cloudinary.com', 'via.placeholder.com'],
    formats: ['image/webp', 'image/avif'],
  },
  
  // Production optimizations
  compress: true,
  poweredByHeader: false,
  output: process.env.NODE_ENV === 'production' ? 'standalone' : undefined,
  
  transpilePackages: ['firebase'],
  
  webpack: (config, { isServer }) => {
    // Resolve issues with private class fields in undici
    if (!isServer) {
      config.resolve.alias = {
        ...config.resolve.alias,
        // Avoid issues with undici's use of private class fields
        undici: false,
      };
      
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
      };
    }
    
    // Production optimizations
    if (process.env.NODE_ENV === 'production') {
      config.optimization.splitChunks = {
        chunks: 'all',
        cacheGroups: {
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            chunks: 'all',
          },
        },
      };
    }
    
    return config;
  },
  
  // Security headers
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
        ],
      },
    ];
  },
  
  // API rewrites
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
  
  // Development optimizations (only in dev mode)
  ...(process.env.NODE_ENV === 'development' && {
    onDemandEntries: {
      // Keep pages in memory for longer during development
      maxInactiveAge: 25 * 1000,
      pagesBufferLength: 2,
    },
  }),
};

module.exports = nextConfig;
