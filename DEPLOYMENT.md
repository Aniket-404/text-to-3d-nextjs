# Deployment Instructions

This document provides instructions for deploying the Text-to-3D Converter application.

## Prerequisites

- Node.js 18.x or higher
- Python 3.9 or higher
- Firebase account
- Cloudinary account
- Hugging Face account (for the Stable Diffusion API)

## Environment Setup

1. Set up the required environment variables in `.env.local`:

```
# Firebase
NEXT_PUBLIC_FIREBASE_API_KEY=your-api-key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your-app.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your-project-id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your-app.appspot.com
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your-sender-id
NEXT_PUBLIC_FIREBASE_APP_ID=your-app-id

# Cloudinary
NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret

# API URL (for production)
NEXT_PUBLIC_API_URL=https://your-python-api-url
```

2. Set up the Python API environment variables in `python-api/.env`:

```
# Hugging Face API Key
HUGGINGFACE_API_KEY=your-huggingface-api-key

# Stable Diffusion API Endpoint
SD_API_URL=https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2
```

## Frontend Deployment

1. Build the Next.js application:

```bash
npm run build
```

2. Deploy to Vercel, Netlify, or your preferred hosting provider.

## Python API Deployment

1. Install the Python dependencies:

```bash
cd python-api
pip install -r requirements.txt
```

2. Deploy the Flask application to a service like Heroku, AWS, or Google Cloud.

3. Update the `NEXT_PUBLIC_API_URL` environment variable in the frontend to point to your deployed Python API.

## Post-Deployment Steps

1. Configure Firebase Authentication to enable email/password sign-in.
2. Set up Firebase security rules for Firestore to protect user data.
3. Test the application by creating a user account and generating a 3D model.

## Troubleshooting

- If you encounter CORS issues, ensure your API is properly configured to accept requests from your frontend domain.
- For Firebase authentication issues, check the Firebase console for error logs.
- For model generation issues, check the Python API logs for details.
