# Text to 3D Model Converter

A web application that converts text descriptions into 3D models using AI.

## Features

- Convert text prompts into 3D models
- View and interact with 3D models in the browser
- User authentication and model storage
- Download and share generated models

## Tech Stack

- **Frontend**: Next.js, TypeScript, Tailwind CSS, Three.js (React Three Fiber)
- **Backend**: Python Flask API, Hugging Face Stable Diffusion
- **Authentication**: Firebase Authentication
- **Database**: Firebase Firestore
- **Storage**: Cloudinary

## Getting Started

### Prerequisites

- Node.js 18.x or higher
- Python 3.9 or higher
- Firebase account
- Cloudinary account
- Hugging Face account

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/text-to-3d-converter.git
cd text-to-3d-converter
```

2. Install frontend dependencies:

```bash
npm install
```

3. Install Python API dependencies:

```bash
cd python-api
pip install -r requirements.txt
cd ..
```

4. Create environment variables:

   - Create a `.env.local` file in the root directory for Next.js
   - Create a `.env` file in the `python-api` directory for the Flask API

5. Start the development servers:

```bash
# Start the Next.js frontend
npm run dev

# In a separate terminal, start the Python API
cd python-api
python app.py
```

6. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Usage

1. Sign in or create an account
2. Enter a text prompt describing the 3D model you want to create
3. Wait for the AI to generate your model
4. View, interact with, and download your 3D model
5. Access your previously created models from the dashboard

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [React Three Fiber](https://github.com/pmndrs/react-three-fiber) for 3D rendering
- [Hugging Face](https://huggingface.co/) for AI models
- [Firebase](https://firebase.google.com/) for authentication and database
- [Cloudinary](https://cloudinary.com/) for media storage
- [Tailwind CSS](https://tailwindcss.com/) for styling
