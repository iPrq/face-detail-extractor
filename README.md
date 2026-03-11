# Face Detail Extractor

Face Detail Extractor is a multi-task deep learning application designed to estimate age and classify gender and ethnicity from facial images. Built with Keras 3 and the ResNet-50 architecture, the system leverages transfer learning and fine-tuning on the UTKFace dataset to provide real-time facial analysis through a FastAPI backend and a modern Next.js frontend.

## Features

* **Multi-Task Learning**: Predicts three distinct facial attributes (Age, Gender, and Ethnicity) simultaneously from a single image pass.
* **Deep Learning Backbone**: Utilizes a ResNet-50 backbone pretrained on ImageNet for robust feature extraction.
* **Dual-Phase Training**: Implements a structured training approach involving initial frozen-backbone training followed by full-model fine-tuning with a reduced learning rate.
* **Production-Ready API**: Features a FastAPI backend with automated image preprocessing and inference handling.
* **Modern Interface**: Includes a responsive Next.js frontend with drag-and-drop support and real-time confidence scoring.

## Technical Architecture

### Model Design

The architecture consists of a shared ResNet-50 feature extractor followed by three specialized output heads:

* **Age Head**: A regression branch utilizing Mean Squared Error (MSE) loss and Mean Absolute Error (MAE) metrics.
* **Gender Head**: A binary classification branch utilizing Binary Crossentropy loss.
* **Ethnicity Head**: A categorical classification branch utilizing Sparse Categorical Crossentropy loss for five distinct ethnic categories.

### Dataset

The model is trained on the UTKFace dataset, consisting of over 20,000 images with annotations for age, gender, and race.

## Installation and Setup

### Prerequisites

* Python 3.10 or higher
* Node.js 18.x or higher
* TensorFlow or a compatible Keras 3 backend

### Backend Setup

1. Navigate to the backend directory.
2. Install the required Python dependencies:
```bash
pip install fastapi uvicorn keras keras_hub tensorflow-cpu pillow numpy python-multipart

```


3. Ensure your trained model file (`face_multi_task.keras`) is located in the root of the backend directory.
4. Start the FastAPI server:
```bash
python main.py

```


The API will be available at `http://localhost:8000`.

### Frontend Setup

1. Navigate to the frontend directory.
2. Install the Node dependencies:
```bash
npm install

```


3. Start the development server:
```bash
npm run dev

```


The interface will be available at `http://localhost:3000`.

## Deployment

### Hugging Face Spaces

This project is configured for deployment on Hugging Face Spaces using the Docker SDK.

1. Create a new Space with the Docker SDK.
2. Upload `main.py`, `Dockerfile`, `requirements.txt`, and the model file.
3. Due to the model file size (~277 MB), ensure Git LFS is initialized if deploying via CLI:
```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add face_multi_task.keras
git commit -m "Upload model via LFS"
git push

```



## License

This project is licensed under the MIT License.
