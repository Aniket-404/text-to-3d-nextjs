'use client';

import { useState, useEffect } from 'react';
import { 
  User,
  Auth,
  onAuthStateChanged 
} from 'firebase/auth';
import { getFirebaseAuth } from '@/lib/firebase';

interface UseAuthReturn {
  user: User | null;
  loading: boolean;
  error: string | null;
  signUp: (email: string, password: string) => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
}

export default function useAuth(): UseAuthReturn {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [auth, setAuth] = useState<Auth | null>(null);

  // Initialize auth on component mount (client-side only)
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        const authInstance = await getFirebaseAuth();
        setAuth(authInstance);
      } catch (err) {
        console.error('Failed to initialize auth:', err);
        setError('Failed to initialize authentication');
        setLoading(false);
      }
    };

    if (typeof window !== 'undefined') {
      initializeAuth();
    }
  }, []);

  // Set up auth state listener once auth is initialized
  useEffect(() => {
    if (!auth) return;
    
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        setUser(user);
      } else {
        setUser(null);
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, [auth]);

  const signUp = async (email: string, password: string) => {
    setLoading(true);
    setError(null);
    
    try {
      if (!auth) {
        throw new Error('Authentication not initialized');
      }
      
      // Dynamically import Firebase auth functions to avoid SSR issues
      const { createUserWithEmailAndPassword } = await import('firebase/auth');
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      setUser(userCredential.user);
    } catch (error: any) {
      console.error('Sign up error:', error);
      setError(error.message || 'Failed to sign up');
    } finally {
      setLoading(false);
    }
  };

  const signIn = async (email: string, password: string) => {
    setLoading(true);
    setError(null);
    
    try {
      if (!auth) {
        throw new Error('Authentication not initialized');
      }
      
      // Dynamically import Firebase auth functions to avoid SSR issues
      const { signInWithEmailAndPassword } = await import('firebase/auth');
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      setUser(userCredential.user);
    } catch (error: any) {
      console.error('Sign in error:', error);
      setError(error.message || 'Failed to sign in');
    } finally {
      setLoading(false);
    }
  };

  const signOut = async () => {
    setLoading(true);
    
    try {
      if (!auth) {
        throw new Error('Authentication not initialized');
      }
      
      // Dynamically import Firebase auth functions to avoid SSR issues
      const { signOut: firebaseSignOut } = await import('firebase/auth');
      await firebaseSignOut(auth);
      setUser(null);
    } catch (error: any) {
      console.error('Sign out error:', error);
      setError(error.message || 'Failed to sign out');
    } finally {
      setLoading(false);
    }
  };

  return {
    user,
    loading,
    error,
    signUp,
    signIn,
    signOut,
  };
}
