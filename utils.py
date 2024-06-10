# # import numpy as np
# # import cv2
# # from sklearn.cluster import KMeans


# # def process_image(image):
# #     # Convert to grayscale
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     # Apply Canny Edge Detection
# #     edges = cv2.Canny(gray, 50, 150)
# #     # Apply K-Means Clustering
# #     pixel_values = image.reshape((-1, 3))
# #     pixel_values = np.float32(pixel_values)

# #     k = 2
# #     kmeans = KMeans(n_clusters=k, random_state=0)
# #     kmeans.fit(pixel_values)
# #     centers = np.uint8(kmeans.cluster_centers_)
# #     labels = kmeans.labels_
# #     segmented_image = centers[labels.flatten()]
# #     segmented_image = segmented_image.reshape(image.shape)

# #     return segmented_image


# # def calculate_foot_length(image):
# #     # Assuming the paper size (A4) is 29.7 cm x 21.0 cm
# #     h, w = image.shape[:2]
# #     paper_width_cm = 21.0

# #     # Find contours
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     if len(contours) == 0:
# #         return 0
# #     # Assuming the largest contour is the foot
# #     foot_contour = max(contours, key=cv2.contourArea)
# #     x, y, width, height = cv2.boundingRect(foot_contour)

# #     # Convert width in pixels to cm
# #     foot_length_cm = (width / w) * paper_width_cm
# #     return foot_length_cm


# # def recommend_shoe_size(foot_length, gender):
# #     if gender == "Male":
# #         # Size conversion based on foot length for men (example values, adjust as necessary)
# #         ue_size = int(foot_length * 1.5)
# #         us_size = int(foot_length * 1.4)
# #         uk_size = int(foot_length * 1.3)
# #     else:
# #         # Size conversion based on foot length for women (example values, adjust as necessary)
# #         ue_size = int(foot_length * 1.4)
# #         us_size = int(foot_length * 1.3)
# #         uk_size = int(foot_length * 1.2)

# #     return ue_size, us_size, uk_size

# #IMPORT LIBRARY
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib_inline
# from scipy import ndimage
# from imutils import contours
# import argparse
# import imutils
# import cv2
# from sklearn.cluster import KMeans
# import random as rng

# #Image preprocessing
# def preprocess(img):

#     img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

#     img = cv2.GaussianBlur(img, (9, 9), 0)
#     img = img/255

#     return img

# def edgeDetection(clusteredImage):
#   edged1 = cv2.Canny(clusteredImage, 0, 255)
#   edged = cv2.dilate(edged1, None, iterations=1)
#   edged = cv2.erode(edged, None, iterations=1)
#   return edged

# #crop kertas A4
# #Fungsi ini memotong citra asli (oimg) berdasarkan bounding rectangle (bRect) yang diperoleh dari deteksi tepi.
# #Proses pemotongan dilakukan dengan mengambil sebagian dari citra asli sesuai dengan koordinat (x, y) dan lebar serta tinggi bounding rectangle.
# def cropOrig(bRect, oimg):
#     # x (Horizontal), y (Vertical Downwards) are start coordinates
#     # img.shape[0] = height of image
#     # img.shape[1] = width of image

#     x,y,w,h = bRect

#     print(x,y,w,h)
#     pcropedImg = oimg[y:y+h,x:x+w]

#     x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]

#     y2 = int(h1/10)

#     x2 = int(w1/10)

#     crop1 = pcropedImg[y1+y2:h1-y2,x1+x2:w1-x2]

#     ix, iy, iw, ih = x+x2, y+y2, crop1.shape[1], crop1.shape[0]

#     croppedImg = oimg[iy:iy+ih,ix:ix+iw]

#     return croppedImg, pcropedImg

# #Processing gambar setelah dicrop
# def overlayImage(croppedImg, pcropedImg):


#     x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]

#     y2 = int(h1/10)

#     x2 = int(w1/10)

#     new_image = np.zeros((pcropedImg.shape[0], pcropedImg.shape[1], 3), np.uint8)
#     new_image[:, 0:pcropedImg.shape[1]] = (255, 0, 0) # (B, G, R)

#     new_image[ y1+y2:y1+y2+croppedImg.shape[0], x1+x2:x1+x2+croppedImg.shape[1]] = croppedImg

#     return new_image

# #membuat bounding box
# def getBoundingBox(img):

#     contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

#     contours_poly = [None]*len(contours)
#     boundRect = [None]*len(contours)

#     for i, c in enumerate(contours):
#         contours_poly[i] = cv2.approxPolyDP(c, 3, True)
#         boundRect[i] = cv2.boundingRect(contours_poly[i])


#     return boundRect, contours, contours_poly, img

# def drawCnt(bRect, contours, cntPoly, img):

#     drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)


#     paperbb = bRect

#     for i in range(len(contours)):
#       color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
#       cv2.drawContours(drawing, cntPoly, i, color)
#     cv2.rectangle(drawing, (int(paperbb[0]), int(paperbb[1])), \
#               (int(paperbb[0]+paperbb[2]), int(paperbb[1]+paperbb[3])), color, 2)

#     return drawing

# gender = input("Input gender anda (pria/wainta) : ")

# oimg = cv2.imread('/content/Kaki4.jpg')

# preprocessedOimg = preprocess(oimg)
# plt.imshow(preprocessedOimg)
# plt.show()

# # For clustering the image using k-means, we first need to convert it into a 2-dimensional array
# # (H*W, N) N is channel = 3
# image_2D = preprocessedOimg.reshape(preprocessedOimg.shape[0]*preprocessedOimg.shape[1], preprocessedOimg.shape[2])

# kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2D)
# clustOut = kmeans.cluster_centers_[kmeans.labels_]

# # Reshape back the image from 2D to 3D image
# clustered_3D = clustOut.reshape(preprocessedOimg.shape[0], preprocessedOimg.shape[1], preprocessedOimg.shape[2])

# clusteredImg = np.uint8(clustered_3D*255)
# plt.imshow(clusteredImg)
# plt.show()

# edgedImg = edgeDetection(clusteredImg)
# plt.imshow(edgedImg)
# plt.show()

# boundRect, contours, contours_poly, img = getBoundingBox(edgedImg)
# pdraw = drawCnt(boundRect[1], contours, contours_poly, img)
# plt.imshow(pdraw)
# plt.show()

# croppedImg, pcropedImg = cropOrig(boundRect[1], clusteredImg)
# plt.imshow(croppedImg)
# plt.show()
# plt.imshow(pcropedImg)
# plt.show()

# newImg = overlayImage(croppedImg, pcropedImg)
# x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]

# y2 = int(h1/10)

# x2 = int(w1/10)

# new_image = np.zeros((pcropedImg.shape[0], pcropedImg.shape[1], 3), np.uint8)
# new_image[:, 0:pcropedImg.shape[1]] = (255, 0, 0) # (B, G, R)

# new_image[ y1+y2:y1+y2+croppedImg.shape[0], x1+x2:x1+x2+croppedImg.shape[1]] = croppedImg

# plt.imshow(new_image)
# plt.show()

# fedged = edgeDetection(new_image)

# fboundRect, fcnt, fcntpoly, fimg = getBoundingBox(fedged)
# fdraw = drawCnt(fboundRect[2], fcnt, fcntpoly, fimg)
# plt.imshow(fdraw)
# plt.show()

# x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]

# y2 = int(h1/10)

# x2 = int(w1/10)

# fh = y2 + fboundRect[2][3]
# fw = x2 + fboundRect[2][2]
# ph = pcropedImg.shape[0]
# pw = pcropedImg.shape[1]

# #print("Feet height: ", fh)
# #print("Feet Width: ", fw)

# #print("Paper height: ", ph)
# #print("Paper Width: ", pw)

# opw = 21
# oph = 29.7

# ofs = 0.0


# if fw>fh:
#   ofs = (oph/pw)*fw
# else :
#   ofs = (oph/ph)*fh


# print("Panjang kaki anda(cm)   :", ofs)

# if gender == "Wanita" or "wanita" :
#   if ofs >=21.35 and ofs <=22.02 :
#     print("Ukuran kaki anda adalah : 36")
#   elif ofs >=22.02 and ofs <=22.69 :
#     print("Ukuran kaki anda adalah : 37")
#   elif ofs >=22.69 and ofs <=23.36 :
#     print("Ukuran kaki anda adalah : 38")
#   elif ofs >=23.36 and ofs <=24.03 :
#     print("Ukuran kaki anda adalah : 39")
#   elif ofs >=24.03 and ofs <=24.70 :
#     print("Ukuran kaki anda adalah : 40")
#   elif ofs >=24.70 and ofs <=25.37 :
#     print("Ukuran kaki anda adalah : 41")
#   elif ofs >=25.37 and ofs <=26.04 :
#     print("Ukuran kaki anda adalah : 42")
#   elif ofs >=26.04 and ofs <=26.71 :
#     print("Ukuran kaki anda adalah : 43")

# elif gender == "Pria" or "pria" :
#   if ofs >=21.35 and ofs <=22.02 :
#     print("Ukuran kaki anda adalah : 37")
#   elif ofs >=22.02 and ofs <=22.69 :
#     print("Ukuran kaki anda adalah : 38")
#   elif ofs >=22.69 and ofs <=23.36 :
#     print("Ukuran kaki anda adalah : 39")
#   elif ofs >=23.36 and ofs <=24.03 :
#     print("Ukuran kaki anda adalah : 40")
#   elif ofs >=24.03 and ofs <=24.70 :
#     print("Ukuran kaki anda adalah : 41")
#   elif ofs >=24.70 and ofs <=25.37 :
#     print("Ukuran kaki anda adalah : 42")
#   elif ofs >=25.37 and ofs <=26.04 :
#     print("Ukuran kaki anda adalah : 43")
#   elif ofs >=26.04 and ofs <=26.71 :
#     print("Ukuran kaki anda adalah : 44")

import numpy as np
import cv2
import random as rng
from sklearn.cluster import KMeans


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = img / 255
    return img


def edgeDetection(clusteredImage):
    edged1 = cv2.Canny(clusteredImage, 0, 255)
    edged = cv2.dilate(edged1, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    return edged


def cropOrig(bRect, oimg):
    x, y, w, h = bRect
    pcropedImg = oimg[y : y + h, x : x + w]
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
    y2 = int(h1 / 10)
    x2 = int(w1 / 10)
    crop1 = pcropedImg[y1 + y2 : h1 - y2, x1 + x2 : w1 - x2]
    ix, iy, iw, ih = x + x2, y + y2, crop1.shape[1], crop1.shape[0]
    croppedImg = oimg[iy : iy + ih, ix : ix + iw]
    return croppedImg, pcropedImg


def overlayImage(croppedImg, pcropedImg):
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
    y2 = int(h1 / 10)
    x2 = int(w1 / 10)
    new_image = np.zeros((pcropedImg.shape[0], pcropedImg.shape[1], 3), np.uint8)
    new_image[:, 0 : pcropedImg.shape[1]] = (255, 0, 0)
    new_image[
        y1 + y2 : y1 + y2 + croppedImg.shape[0], x1 + x2 : x1 + x2 + croppedImg.shape[1]
    ] = croppedImg
    return new_image


def getBoundingBox(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    return boundRect, contours, contours_poly, img


def drawCnt(bRect, contours, cntPoly, img):
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    paperbb = bRect
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(drawing, cntPoly, i, color)
        cv2.rectangle(
            drawing,
            (int(paperbb[0]), int(paperbb[1])),
            (int(paperbb[0] + paperbb[2]), int(paperbb[1] + paperbb[3])),
            color,
            2,
        )
    return drawing


def getShoeSize(ofs, gender):
    if gender.lower() == "wanita":
        if 21.35 <= ofs <= 22.02:
            return "36"
        elif 22.02 <= ofs <= 22.69:
            return "37"
        elif 22.69 <= ofs <= 23.36:
            return "38"
        elif 23.36 <= ofs <= 24.03:
            return "39"
        elif 24.03 <= ofs <= 24.70:
            return "40"
        elif 24.70 <= ofs <= 25.37:
            return "41"
        elif 25.37 <= ofs <= 26.04:
            return "42"
        elif 26.04 <= ofs <= 26.71:
            return "43"
    elif gender.lower() == "pria":
        if 21.35 <= ofs <= 22.02:
            return "37"
        elif 22.02 <= ofs <= 22.69:
            return "38"
        elif 22.69 <= ofs <= 23.36:
            return "39"
        elif 23.36 <= ofs <= 24.03:
            return "40"
        elif 24.03 <= ofs <= 24.70:
            return "41"
        elif 24.70 <= ofs <= 25.37:
            return "42"
        elif 25.37 <= ofs <= 26.04:
            return "43"
        elif 26.04 <= ofs <= 26.71:
            return "44"
    return "Ukuran tidak ditemukan"
