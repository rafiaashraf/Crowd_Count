import tensorflow as tf
import numpy as np
import os
import cv2
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from pyngrok import ngrok
import matplotlib.pyplot as plt

upload_folder = "../static/container"

app = Flask(__name__,
            template_folder="../templates",
            static_folder="../static")
app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = upload_folder
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Create a tunnel to expose the app
public_url = ngrok.connect(5000)
print(' * Public URL:', public_url)

extensions = set(["png", "jpg", "jpeg"])


def extension(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions


def preprocess_image(file_path, target_size=(640, 480)):
    """Preprocess the image for prediction."""
    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, target_size)  # Resize to match model input size
    img_normalized = img_resized / 255.0  # Normalize
    img_expanded = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
    return img_expanded


def density_model(filename):
    """Load the model and perform prediction on the uploaded image."""
    # Load the pre-trained model
    model = tf.keras.models.load_model("crowd_count_trained.h5")

    # Prepare the image
    file_path = os.path.join(upload_folder, filename)
    input_image = preprocess_image(file_path)

    # Predict density map
    predicted_density_map = model.predict(input_image).squeeze()

    # Calculate the head count
    scale = 10000.0  # Ensure this matches your training normalization scale
    predicted_count = np.sum(predicted_density_map)/scale

    # Save the density map as an image for display
    density_map_path = os.path.join(upload_folder, f"density_{filename}")
    plt.imsave(density_map_path, predicted_density_map, cmap='jet')

    return predicted_count, f"density_{filename}"


@app.route("/")
def upload_form():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    if file.filename == "":
        flash("No image selected for uploading")
        return render_template("home.html")
    if file and extension(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Predict using the model
        predicted_count, density_map_filename = density_model(filename)

        return render_template(
            "home.html",
            filename=filename,
            density_map=density_map_filename,
            res=("The number of people in the picture is around " + str(int(round(predicted_count))))
        )
    else:
        flash("Allowed image types are: png, jpg, jpeg")
        return render_template("home.html")


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for("static", filename="container/" + filename), code=301)


if __name__ == "__main__":
    app.run()
