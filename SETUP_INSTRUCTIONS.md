# Setup Instructions

This document provides detailed setup instructions for developers working on the Text-to-3D Converter project.

## Environment Variables

This project uses several environment variables that need to be set up. For security reasons, these are not checked into version control.

1. Copy the example environment files:
```bash
cp .env.example .env.local
cp python-api/.env.example python-api/.env
```

2. Fill in your credentials in both `.env.local` and `python-api/.env`. Never commit these files to git!

3. If you accidentally commit sensitive data, follow these steps to remove it:
   ```bash
   # Remove file from git history
   git filter-branch --force --index-filter "git rm --cached --ignore-unmatch path/to/file" --prune-empty --tag-name-filter cat -- --all
   
   # Force push to remove sensitive data from remote
   git push origin --force --all
   ```

## Frontend Setup

1. Install Node.js dependencies:

```bash
npm install
```

2. Create a `.env.local` file in the root directory with the following variables:

```
# Firebase
NEXT_PUBLIC_FIREBASE_API_KEY=
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=
NEXT_PUBLIC_FIREBASE_PROJECT_ID=
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=
NEXT_PUBLIC_FIREBASE_APP_ID=

# Cloudinary
NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=

# API URL (for local development)
NEXT_PUBLIC_API_URL=http://localhost:5000
```

3. Start the Next.js development server:

```bash
npm run dev
```

## Python API Setup

1. Navigate to the `python-api` directory:

```bash
cd python-api
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the `python-api` directory with the following variables:

```
# Hugging Face API Key
HUGGINGFACE_API_KEY=

# Stable Diffusion API Endpoint
SD_API_URL=https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2
```

5. Start the Flask API:

```bash
python app.py
```

## Firebase Setup

1. Create a new Firebase project at [firebase.google.com](https://firebase.google.com/).

2. Enable Authentication with Email/Password sign-in method.

3. Create a Firestore database and set up the following security rules:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /models/{modelId} {
      allow read, write: if request.auth != null && request.auth.uid == resource.data.userId;
      allow create: if request.auth != null && request.resource.data.userId == request.auth.uid;
    }
  }
}
```

4. Get your Firebase config values from the Firebase console and add them to your `.env.local` file.

## Cloudinary Setup

1. Create a Cloudinary account at [cloudinary.com](https://cloudinary.com/).

2. Get your Cloudinary credentials from the dashboard and add them to your `.env.local` file.

3. Create an upload preset named `text-to-3d` with the following settings:
   - Unsigned uploading: Enabled
   - Folder: `text-to-3d`

## Hugging Face Setup

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co/).

2. Generate an API key from your account settings.

3. Add your API key to the `python-api/.env` file.

## Testing the Application

1. Ensure both the Next.js frontend and Python API are running.

2. Open [http://localhost:3000](http://localhost:3000) in your browser.

3. Create an account or sign in.

4. Enter a text prompt and test the 3D model generation.
