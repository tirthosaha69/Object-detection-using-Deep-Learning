from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model("Model/cifar10_resnet50.h5")

labels_dictionary = {
    0:'airplane', 1:'automobile', 2:'bird', 3:'cat',
    4:'deer', 5:'dog', 6:'frog', 7:'horse',
    8:'ship', 9:'truck'
}

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(filepath):
    try:
        with Image.open(filepath) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((32, 32), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

try:
    xtest = np.load("xtest.npy")
    ytest = np.load("ytest.npy")
    _, model_accuracy = model.evaluate(xtest, ytest, verbose=0)
    model_accuracy = round(model_accuracy * 100, 2)
except Exception as e:
    print(f"Could not load test data: {e}")
    model_accuracy = None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None
    error_message = None

    if request.method == "POST":
        if "file" not in request.files:
            error_message = "No file selected"
        else:
            file = request.files["file"]
            if file.filename == "":
                error_message = "No file selected"
            elif not allowed_file(file.filename):
                error_message = "Invalid file type. Please upload an image file."
            else:
                try:
                    file_ext = file.filename.rsplit('.', 1)[1].lower()
                    unique_filename = f"{uuid.uuid4().hex}.{file_ext}"
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename).replace("\\", "/")
                    file.save(filepath)
                    img_array = process_image(filepath)
                    if img_array is not None:
                        try:
                            pred = model.predict(img_array, verbose=0)
                            pred_class = np.argmax(pred)
                            confidence = np.max(pred) * 100
                            prediction = labels_dictionary[pred_class]
                            confidence_value = round(confidence, 2)
                            image_path = filepath
                            print(f"Prediction: {prediction} with {confidence_value}% confidence")
                        except Exception as e:
                            error_message = f"Error making prediction: {str(e)}"
                            print(f"Prediction error: {e}")
                    else:
                        error_message = "Could not process the uploaded image"
                except Exception as e:
                    error_message = f"Error processing file: {str(e)}"
                    print(f"File processing error: {e}")

    return render_template("index.html", 
                         prediction=prediction, 
                         image_path=image_path, 
                         accuracy=model_accuracy,
                         error_message=error_message)

@app.errorhandler(413)
def too_large(e):
    return render_template("index.html", 
                         error_message="File too large. Maximum size is 16MB."), 413

if __name__ == "__main__":
    app.run(debug=True)
