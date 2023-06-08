from flask import Flask, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/recognize', methods=['POST'])
def recognize_image():
    # Access the image data sent from the frontend
    image_data = request.form.get('image')

    # Save the image data to a file (e.g., specific_image.jpg)
    with open('specific_image.jpg', 'wb') as img_file:
        img_file.write(image_data.decode('base64'))

    # Perform the image recognition using the saved image file

    # Load the trained model
    model = tf.keras.models.load_model('custom_model.h5')

    # Path to the specific image you want to recognize
    specific_image_path = 'images/qr.jpg'

    # Load the specific image and preprocess it
    img = image.load_img(specific_image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Perform inference on the specific image
    preds = model.predict(x)
    class_index = np.argmax(preds[0])
    class_label = decode_predictions(preds, top=1)[0][0][1]

    # Check if the specific image is recognized
    if class_label == 'specific_object':
        return 'specific_object'
    else:
        return 'not_specific_object'

if __name__ == '__main__':
    app.run()
