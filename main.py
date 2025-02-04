from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = ["*"]  # Allow all origins for testing purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model with tf.saved_model.load
MODEL = tf.saved_model.load("model")

CLASS_NAMES = ['Banana Black Sigatoka Disease',
               'Banana Bract Mosaic Virus Disease',
               'Banana Healthy Leaf',
               'Banana Insect Pest Disease',
               'Banana Moko Disease',
               'Banana Panama Disease',
               'Banana Yellow Sigatoka Disease',
               'LAYU FUSARIUM']


@app.get("/")
async def home():
    return "WELCOME TO BANANA DISEASE API BY PANCA"


@app.get("/ping")
async def ping():
    return "API is working fine"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Resize the image to the expected size (500, 500)
    image = Image.fromarray(image).resize((500, 500))
    # Normalize the image to [0, 1] and convert to float32
    image = np.array(image).astype(np.float32) / 255.0
    return image


@app.post("/test")
async def test(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predicted_class = "TESTING PREDICT"
    confidence = .98
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    try:
        # Read and preprocess the image
        image = read_file_as_image(await file.read())
        image = preprocess_image(image)

        # Add a batch dimension
        img_batch = np.expand_dims(image, 0)

        # Ensure img_batch is a float32 tensor
        img_batch = tf.convert_to_tensor(img_batch, dtype=tf.float32)

        # Make predictions using the model
        predictions = MODEL(img_batch, training=False).numpy()

        # Extract the predicted class and confidence
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=443)
