import cv2 as cv
import numpy as np

img_fn = ["images/img3.JPG", "images/img2.JPG", "images/img1.JPG", "images/img0.JPG"]

# Load and align images
img_list = [cv.imread(fn) for fn in img_fn]
# alignMBT = cv.createAlignMTB()
# alignMBT.process(img_list, img_list)

# Merge images into HDR file
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)
merge_debevec = cv.createMergeDebevec()
hdr_img = merge_debevec.process(img_list, times=exposure_times.copy())

# Mantiuk tonemap
# Params : gamma, scale, saturation
tonemapMantiuk = cv.createTonemapMantiuk(2.2,0.85, 1.2)
ldrMantiuk = tonemapMantiuk.process(hdr_img)
ldrMantiuk_8bit = np.clip(ldrMantiuk*255, 0, 255).astype('uint8')
cv.imwrite("images/res_mantiuk.jpg", ldrMantiuk_8bit)
cv.imshow("Mantiuk", ldrMantiuk_8bit)

img_ldr = cv.imread(image/img2)
