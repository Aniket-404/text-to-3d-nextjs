# 3Dify - Text to 3D Model Converter

A production-ready web application that converts text descriptions and images into high-quality 3D models using advanced AI and Neural Radiance Fields (NeRF).

## 🚀 Features

### **Core Generation Modes**
- **Text to 3D**: Convert any text description into a 3D model
- **Image to 3D**: Upload images and convert them to 3D models
- **Dual Quality Options**: Fast preview + Premium NeRF generation

### **Advanced Technology**
- **Neural Radiance Fields (NeRF)**: State-of-the-art 3D reconstruction
- **Multi-View Synthesis**: Generate models from multiple camera angles
- **Real-time Progress Tracking**: Live updates during generation
- **Interactive 3D Viewer**: Web-based model exploration with Three.js

### **Production Features**
- **User Authentication**: Secure Firebase-based auth system
- **Model Storage**: Persistent storage with Firestore database
- **Asset Management**: Cloudinary integration for scalable file handling
- **Multiple Export Formats**: OBJ meshes, NeRF weights, config files
- **Responsive Design**: Modern UI that works on all devices

## 🛠 Tech Stack

- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, React Three Fiber
- **Backend**: Python Flask API with production NeRF pipeline
- **AI/ML**: Stable Diffusion, PyTorch, Intel DPT, Open3D
- **Authentication**: Firebase Authentication
- **Database**: Firebase Firestore
- **Storage**: Cloudinary for scalable asset management
- **3D Processing**: Neural Radiance Fields, Marching Cubes, Mesh Optimization

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
