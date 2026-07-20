import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_cropper import st_cropper

# Load Model
model = load_model("face-mask-detector.keras")

# Title
st.title("Face Mask Detection Project")

# Select Input
option = st.selectbox("Select Input", ["Image", "Capture"])

uploaded_file = None
camera_image = None
cropped_img = None

# ---------------- Upload Image ----------------
if option == "Image":
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

# ---------------- Camera ----------------
else:
    camera_image = st.camera_input("Capture Image")

    if camera_image is not None:
        image = Image.open(camera_image)

        st.write("Crop Image")

        cropped_img = st_cropper(
            image,
            realtime_update=True,
            box_color="red",
            aspect_ratio=(1, 1)
        )

        st.image(cropped_img, caption="Cropped Image", width=300)

# ---------------- Detect ----------------
if st.button("Detect"):

    image_to_detect = None

    if uploaded_file is not None:
        image_to_detect = Image.open(uploaded_file)

    elif cropped_img is not None:
        image_to_detect = cropped_img

    else:
        st.error("Please upload or capture an image.")

    if image_to_detect is not None:

        # Show Uploaded Image
        st.image(
            image_to_detect,
            caption="Uploaded Image",
            width=350
        )

        # Preprocess Image
        img = image_to_detect.convert("RGB")
        img = img.resize((150, 150))

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array, verbose=0)

        score = float(prediction[0][0])

        st.write(f"### Prediction Score : {score:.4f}")

        # Result
        if score <= 0.5:
            st.success("Person is WITH MASK")
        else:
            st.error("Person is WITHOUT MASK")
