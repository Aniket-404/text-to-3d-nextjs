'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import useAuth from '@/hooks/useAuth';
import toast from 'react-hot-toast';
import { FaSpinner, FaArrowLeft } from 'react-icons/fa';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [localError, setLocalError] = useState<string | null>(null);
  const { signIn, user, loading, error } = useAuth();
  const router = useRouter();

  // Redirect if already logged in
  useEffect(() => {
    if (user) {
      router.push('/dashboard');
    }
  }, [user, router]);

  // Display firebase auth errors
  useEffect(() => {
    if (error) {
      toast.error(error);
    }
  }, [error]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);
    
    if (!email || !password) {
      setLocalError('Please fill in all fields');
      toast.error('Please fill in all fields');
      return;
    }
    
    try {
      await signIn(email, password);
      // The useEffect above will handle redirection on successful login
    } catch (err: any) {
      console.error('Login error:', err);
      // Auth errors are already handled in the useAuth hook
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-4 bg-gradient-radial from-background to-black">
      <div className="w-full max-w-md p-8 glass-panel">
        <h1 className="text-3xl font-bold text-gradient mb-6 text-center">Sign In</h1>
        
        {(error || localError) && (
          <div className="mb-4 p-3 bg-red-900/50 border border-red-700 text-white rounded">
            {error || localError}
          </div>
        )}
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="email" className="block text-text-primary mb-1">
              Email
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full p-3 rounded input-gradient text-text-primary"
              disabled={loading}
              required
            />
          </div>
          <div>
            <label htmlFor="password" className="block text-text-primary mb-1">
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full p-3 rounded input-gradient text-text-primary"
              disabled={loading}
              required
            />
          </div>
          
          <button
            type="submit"
            disabled={loading}
            className="w-full gradient-button py-3 font-bold hover:shadow-glow transition-all duration-300"
          >
            {loading ? (
              <span className="flex items-center justify-center">
                <FaSpinner className="animate-spin mr-2" /> Signing In...
              </span>
            ) : (
              'Sign In'
            )}
          </button>
        </form>
        
        <div className="mt-6 text-center">
          <p className="text-text-secondary">
            Don&apos;t have an account?{' '}
            <Link href="#" className="text-primary hover:underline">
              Sign Up
            </Link>
          </p>
        </div>
      </div>
      
      <div className="mt-8">
        <Link href="/" className="text-primary hover:underline flex items-center">
          <FaArrowLeft className="mr-2" /> Back to Home
        </Link>
      </div>
    </main>
  );
}
