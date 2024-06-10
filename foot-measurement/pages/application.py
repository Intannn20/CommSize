import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
import utils  # Importing the utility functions


def show():
    st.title("Foot Measurement Application")

    gender = st.selectbox("Select your gender:", ["Pria", "Wanita"])
    uploaded_file = st.file_uploader(
        "Upload an image of your foot on an A4 paper:", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        oimg = cv2.imdecode(file_bytes, 1)

        # Preprocess image
        preprocessedOimg = utils.preprocess(oimg)

        # K-means clustering
        image_2D = preprocessedOimg.reshape(
            preprocessedOimg.shape[0] * preprocessedOimg.shape[1],
            preprocessedOimg.shape[2],
        )
        kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2D)
        clustOut = kmeans.cluster_centers_[kmeans.labels_]
        clustered_3D = clustOut.reshape(
            preprocessedOimg.shape[0],
            preprocessedOimg.shape[1],
            preprocessedOimg.shape[2],
        )
        clusteredImg = np.uint8(clustered_3D * 255)

        # Edge detection
        edgedImg = utils.edgeDetection(clusteredImg)

        # Bounding box
        boundRect, contours, contours_poly, img = utils.getBoundingBox(edgedImg)

        # Crop image
        croppedImg, pcropedImg = utils.cropOrig(boundRect[1], clusteredImg)

        # Overlay image
        newImg = utils.overlayImage(croppedImg, pcropedImg)

        # Final edge detection
        fedged = utils.edgeDetection(newImg)

        # Final bounding box
        fboundRect, fcnt, fcntpoly, fimg = utils.getBoundingBox(fedged)

        # Calculate foot size
        x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
        y2 = int(h1 / 10)
        x2 = int(w1 / 10)
        fh = y2 + fboundRect[2][3]
        fw = x2 + fboundRect[2][2]
        ph = pcropedImg.shape[0]
        pw = pcropedImg.shape[1]
        opw = 21
        oph = 29.7
        ofs = 0.0
        if fw > fh:
            ofs = (oph / pw) * fw
        else:
            ofs = (oph / ph) * fh

        st.write(f"Panjang kaki anda (cm): {ofs:.2f}")

        # Determine shoe size
        shoe_size = utils.getShoeSize(ofs, gender)
        st.write(f"Ukuran kaki anda adalah: {shoe_size}")

        st.image(newImg, caption="Processed Image", use_column_width=True)
