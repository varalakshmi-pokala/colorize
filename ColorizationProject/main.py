import numpy as np
import cv2
from cv2 import dnn
import sys
import os

#--------Model file paths--------#
proto_file = 'Model/colorization_deploy_v2.prototxt'
model_file = 'Model/colorization_release_v2.caffemodel'
hull_pts = 'Model/pts_in_hull.npy'
img_path = 'images/img1.jpg'
#--------------------------------#

print("Program started...")

#--------Reading the model params--------#
net = dnn.readNetFromCaffe(proto_file, model_file)
kernel = np.load(hull_pts)
print("Model loaded OK")

#-----Reading and preprocessing image--------#
img = cv2.imread(img_path)
if img is None:
    print("IMAGE NOT FOUND!")
    sys.exit()
else:
    print("Image loaded successfully")

scaled = img.astype("float32") / 255.0
lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

#-----Add cluster centers to model--------#
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

#-----Resize & prepare L channel--------#
resized = cv2.resize(lab_img, (224, 224))
L = cv2.split(resized)[0]
L -= 50

#-----Predict ab channel--------#
net.setInput(cv2.dnn.blobFromImage(L))
ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

#-----Combine L + predicted ab--------#
L = cv2.split(lab_img)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")

#-----Save output instead of showing--------#
output_path = "output_colorized.jpg"
cv2.imwrite(output_path, colorized)

print("\n✅ OUTPUT SAVED as:", output_path)
print("➡ Open the file manually from your folder.")

