'use client';

import { initializeApp, getApp, getApps } from 'firebase/app';
import { Auth, getAuth } from 'firebase/auth';
import { Firestore, getFirestore } from 'firebase/firestore';

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
  measurementId: process.env.NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID
};

// Initialize Firebase with a safe pattern for Next.js
const getFirebaseApp = () => {
  if (typeof window === 'undefined') {
    // Return null or mock during SSR
    return null;
  }
  
  if (!getApps().length) {
    return initializeApp(firebaseConfig);
  }
  
  return getApp();
};

// Create safe exports that check for client-side before usage
const app = getFirebaseApp();

// These functions should only be called client-side
const getFirebaseAuth = (): Auth | null => {
  const app = getFirebaseApp();
  return app ? getAuth(app) : null;
};

const getFirebaseFirestore = (): Firestore | null => {
  const app = getFirebaseApp();
  return app ? getFirestore(app) : null;
};

// Export the getter functions
export { app, getFirebaseAuth, getFirebaseFirestore };
