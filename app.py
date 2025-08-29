from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load trained model
model = load_model("Model/cifar10_resnet50.h5")

# CIFAR-10 labels
labels_dictionary = {
    0:'airplane', 1:'automobile', 2:'bird', 3:'cat',
    4:'deer', 5:'dog', 6:'frog', 7:'horse',
    8:'ship', 9:'truck'
}

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(filepath):
    """Process uploaded image for model prediction"""
    try:
        # Open and convert image
        with Image.open(filepath) as img:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to CIFAR-10 input size
            img = img.resize((32, 32), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# --- Model Accuracy (run on your test set once) ---
# If you have xtest_scaled and ytest saved, load them here.
# For demo, I'll assume you reloaded them from npy files
try:
    xtest = np.load("xtest.npy")
    ytest = np.load("ytest.npy")
    _, model_accuracy = model.evaluate(xtest, ytest, verbose=0)
    model_accuracy = round(model_accuracy * 100, 2)
except Exception as e:
    print(f"Could not load test data: {e}")
    model_accuracy = None  # If test set not available

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
                    # Generate unique filename to avoid conflicts
                    file_ext = file.filename.rsplit('.', 1)[1].lower()
                    unique_filename = f"{uuid.uuid4().hex}.{file_ext}"
                    
                    # Use forward slashes for cross-platform compatibility
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename).replace("\\", "/")
                    
                    # Save the uploaded file
                    file.save(filepath)
                    
                    # Process the image
                    img_array = process_image(filepath)
                    
                    if img_array is not None:
                        # Make prediction
                        try:
                            pred = model.predict(img_array, verbose=0)
                            pred_class = np.argmax(pred)
                            confidence = np.max(pred) * 100
                            
                            prediction = labels_dictionary[pred_class]
                            confidence_value = round(confidence, 2)
                            
                            # Format the image path for web display
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