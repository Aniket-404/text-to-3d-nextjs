'use client';

import { FaGithub, FaHeart, FaRocket, FaCube, FaBrain, FaDownload } from 'react-icons/fa';

export default function About() {
  return (
    <main className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-10 backdrop-blur-md border-b border-white/5 bg-background/60">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <a href="/" className="text-2xl font-bold text-gradient">
            3Dify
          </a>
          
          <div className="flex items-center space-x-4">
            <a 
              href="/" 
              className="text-text-secondary hover:text-primary transition-colors"
              title="Home"
            >
              Home
            </a>
            <a 
              href="https://github.com/Aniket-404/text-to-3d-nextjs" 
              target="_blank"
              className="text-text-secondary hover:text-primary transition-colors"
              title="GitHub"
            >
              GitHub
            </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-grow container mx-auto px-4 py-10">
        <div className="max-w-4xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <h1 className="text-5xl font-bold mb-6 text-gradient">
              Text to 3D Model Converter
            </h1>
            <p className="text-xl text-text-secondary leading-relaxed">
              A web application that converts text descriptions and images into 3D models using AI. 
              View and interact with 3D models in the browser, then download and share your generated models.
            </p>
          </div>

          {/* Features Section */}
          <div className="glass-panel p-8 mb-8">
            <h2 className="text-3xl font-bold mb-6 flex items-center">
              <FaRocket className="mr-3 text-primary" />
              Features
            </h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <FaBrain className="text-primary mt-1 flex-shrink-0" />
                  <div>
                    <h3 className="font-semibold">AI-Powered Generation</h3>
                    <p className="text-text-secondary text-sm">Convert text prompts into 3D models using advanced AI technology</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <FaCube className="text-primary mt-1 flex-shrink-0" />
                  <div>
                    <h3 className="font-semibold">Image to 3D Conversion</h3>
                    <p className="text-text-secondary text-sm">Upload your own images and convert them to 3D models</p>
                  </div>
                </div>
              </div>
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <FaDownload className="text-primary mt-1 flex-shrink-0" />
                  <div>
                    <h3 className="font-semibold">Download & Share</h3>
                    <p className="text-text-secondary text-sm">Download generated models in OBJ format for use in other applications</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <FaHeart className="text-primary mt-1 flex-shrink-0" />
                  <div>
                    <h3 className="font-semibold">Real-time Progress</h3>
                    <p className="text-text-secondary text-sm">Track generation progress with live updates and detailed status messages</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Tech Stack Section */}
          <div className="glass-panel p-8 mb-8">
            <h2 className="text-3xl font-bold mb-6">
              Tech Stack
            </h2>
            <div className="grid md:grid-cols-3 gap-6">
              <div>
                <h3 className="text-lg font-semibold mb-3 text-primary">Frontend</h3>
                <ul className="space-y-2 text-text-secondary">
                  <li>• Next.js 14</li>
                  <li>• TypeScript</li>
                  <li>• Tailwind CSS</li>
                  <li>• React Three Fiber</li>
                  <li>• React Icons</li>
                </ul>
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-3 text-primary">Backend</h3>
                <ul className="space-y-2 text-text-secondary">
                  <li>• Python Flask API</li>
                  <li>• Hugging Face Stable Diffusion</li>
                  <li>• Intel DPT for Depth</li>
                  <li>• Open3D for 3D Processing</li>
                </ul>
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-3 text-primary">Services</h3>
                <ul className="space-y-2 text-text-secondary">
                  <li>• Cloudinary Storage</li>
                  <li>• Google Gemini AI</li>
                  <li>• Firebase Authentication</li>
                  <li>• Firebase Firestore</li>
                  <li>• Real-time Progress Tracking</li>
                </ul>
              </div>
            </div>
          </div>

          {/* How It Works Section */}
          <div className="glass-panel p-8 mb-8">
            <h2 className="text-3xl font-bold mb-6">
              How It Works
            </h2>
            <div className="space-y-6">
              <div className="flex items-start space-x-4">
                <div className="bg-primary text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">
                  1
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Input Your Idea</h3>
                  <p className="text-text-secondary">Enter a text description or upload an image of what you want to create in 3D</p>
                </div>
              </div>
              <div className="flex items-start space-x-4">
                <div className="bg-primary text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">
                  2
                </div>
                <div>
                  <h3 className="font-semibold mb-2">AI Processing</h3>
                  <p className="text-text-secondary">Our AI generates images (for text) and creates depth maps using advanced computer vision</p>
                </div>
              </div>
              <div className="flex items-start space-x-4">
                <div className="bg-primary text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">
                  3
                </div>
                <div>
                  <h3 className="font-semibold mb-2">3D Model Creation</h3>
                  <p className="text-text-secondary">Point clouds are generated and converted to optimized 3D meshes using Poisson reconstruction</p>
                </div>
              </div>
              <div className="flex items-start space-x-4">
                <div className="bg-primary text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">
                  4
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Download & Use</h3>
                  <p className="text-text-secondary">Get your 3D model in OBJ format, ready for 3D printing, games, or other applications</p>
                </div>
              </div>
            </div>
          </div>

          {/* Getting Started Section */}
          <div className="glass-panel p-8 mb-8">
            <h2 className="text-3xl font-bold mb-6">
              Getting Started
            </h2>
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-3 text-primary">Prerequisites</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <ul className="space-y-2 text-text-secondary">
                    <li>• Node.js 18.x or higher</li>
                    <li>• Python 3.9 or higher</li>
                    <li>• Hugging Face account</li>
                  </ul>
                  <ul className="space-y-2 text-text-secondary">
                    <li>• Cloudinary account</li>
                    <li>• Google Gemini API access</li>
                    <li>• Git for version control</li>
                  </ul>
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold mb-3 text-primary">Installation</h3>
                <div className="space-y-4">
                  <div className="bg-black/20 p-4 rounded-lg">
                    <p className="text-sm text-text-secondary mb-2">1. Clone the repository:</p>
                    <code className="text-primary">git clone https://github.com/yourusername/text-to-3d-converter.git</code>
                  </div>
                  <div className="bg-black/20 p-4 rounded-lg">
                    <p className="text-sm text-text-secondary mb-2">2. Install dependencies:</p>
                    <code className="text-primary">npm install && cd python-api && pip install -r requirements.txt</code>
                  </div>
                  <div className="bg-black/20 p-4 rounded-lg">
                    <p className="text-sm text-text-secondary mb-2">3. Set up environment variables:</p>
                    <code className="text-primary">Create .env.local and python-api/.env files</code>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Usage Section */}
          <div className="glass-panel p-8 mb-8">
            <h2 className="text-3xl font-bold mb-6">
              Usage
            </h2>
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <div className="bg-primary text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0 mt-1">
                  1
                </div>
                <p className="text-text-secondary">Choose between text-to-3D or image-to-3D mode</p>
              </div>
              <div className="flex items-start space-x-3">
                <div className="bg-primary text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0 mt-1">
                  2
                </div>
                <p className="text-text-secondary">Enter a text prompt or upload an image</p>
              </div>
              <div className="flex items-start space-x-3">
                <div className="bg-primary text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0 mt-1">
                  3
                </div>
                <p className="text-text-secondary">Wait for AI processing with real-time progress updates</p>
              </div>
              <div className="flex items-start space-x-3">
                <div className="bg-primary text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold flex-shrink-0 mt-1">
                  4
                </div>
                <p className="text-text-secondary">View, interact with, and download your 3D model</p>
              </div>
            </div>
          </div>

          {/* Open Source Section */}
          <div className="glass-panel p-8 mb-8">
            <h2 className="text-3xl font-bold mb-6 flex items-center">
              <FaGithub className="mr-3 text-primary" />
              Open Source
            </h2>
            <p className="text-text-secondary mb-6 leading-relaxed">
              3Dify is an open-source project built with love for the community. We believe in making AI-powered 3D generation 
              accessible to everyone. The entire codebase is available on GitHub under the MIT License, allowing you to use, 
              modify, and distribute the software freely.
            </p>
            <div className="bg-black/20 p-4 rounded-lg mb-6">
              <p className="text-sm text-text-secondary">
                <strong>License:</strong> MIT License - Feel free to use this project for personal or commercial purposes.
              </p>
            </div>
            <div className="flex flex-wrap gap-4">
              <a
                href="https://github.com/Aniket-404/text-to-3d-nextjs"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 bg-primary hover:bg-primary/80 text-white px-6 py-3 rounded-lg transition-colors"
              >
                <FaGithub />
                <span>View on GitHub</span>
              </a>
            </div>
          </div>

          {/* Acknowledgments */}
          <div className="glass-panel p-8">
            <h2 className="text-3xl font-bold mb-6">
              Acknowledgments
            </h2>
            <p className="text-text-secondary mb-4 leading-relaxed">
              This project wouldn't be possible without these amazing technologies and services:
            </p>
            <div className="grid md:grid-cols-2 gap-4 text-text-secondary">
              <ul className="space-y-2">
                <li>• <strong>React Three Fiber</strong> - 3D rendering in React</li>
                <li>• <strong>Hugging Face</strong> - AI models and infrastructure</li>
                <li>• <strong>Firebase</strong> - Authentication and database</li>
                <li>• <strong>Cloudinary</strong> - Media storage and optimization</li>
                <li>• <strong>Intel DPT</strong> - Depth estimation models</li>
              </ul>
              <ul className="space-y-2">
                <li>• <strong>Open3D</strong> - 3D data processing</li>
                <li>• <strong>Google Gemini</strong> - Advanced AI capabilities</li>
                <li>• <strong>Tailwind CSS</strong> - Beautiful styling framework</li>
                <li>• <strong>Next.js</strong> - React framework</li>
                <li>• <strong>Vercel</strong> - Deployment platform</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <footer className="border-t border-white/5 py-6">
        <div className="container mx-auto px-4 text-center text-text-secondary text-sm">
          <p>© 2025 3Dify - Text to 3D Model Converter</p>
          <p className="mt-1">
            Made with <FaHeart className="inline text-red-500 mx-1" /> by the open source community
          </p>
        </div>
      </footer>
    </main>
  );
}
