import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Toaster } from 'react-hot-toast';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Text to 3D Model Converter',
  description: 'Convert text prompts to 3D models using AI',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <meta name="grammarly-disable" content="true" />
      </head>
      <body className={inter.className}>
        <Toaster
          position="top-center"
          toastOptions={{
            duration: 3000,
            style: {
              background: '#1E1E1E',
              color: '#E0E0E0',
              border: '1px solid rgba(0, 209, 255, 0.3)',
            },
            success: {
              iconTheme: {
                primary: '#00D1FF',
                secondary: '#1E1E1E',
              },
            },
            error: {
              iconTheme: {
                primary: '#FF007A',
                secondary: '#1E1E1E',
              },
            },
          }}
        />
        {children}
      </body>
    </html>
  );
}
