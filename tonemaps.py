from __future__ import division, unicode_literals

import cv2 as cv
import numpy as np

from colour.constants import EPSILON
from colour.models import RGB_COLOURSPACES, RGB_luminance
from colour.utilities import as_float_array

def log_average(a, epsilon=EPSILON):
    a = as_float_array(a)
    average = np.exp(np.average(np.log(a + epsilon)))
    return average

def tonemapping_operator_gamma(RGB, gamma=1, EV=0):
    RGB = as_float_array(RGB)
    exposure = 2 ** EV
    RGB = (exposure * RGB) ** (1.0 / gamma)
    return RGB

def tonemapping_operator_logarithmic(RGB, q=1, k=1, colourspace=RGB_COLOURSPACES['sRGB']):
    RGB = as_float_array(RGB)
    q = 1 if q < 1 else q
    k = 1 if k < 1 else k
    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    L_max = np.max(L)
    L_d = np.log10(1 + L * q) / np.log10(1 + L_max * k)
    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]
    return RGB
    
def tonemapping_operator_exponential(RGB, q=1, k=1, colourspace=RGB_COLOURSPACES['sRGB']):
    RGB = as_float_array(RGB)
    q = 1 if q < 1 else q
    k = 1 if k < 1 else k
    L = RGB_luminance(RGB, colourspace.primaries, colourspace.whitepoint)
    L_a = log_average(L)
    L_d = 1 - np.exp(-(L * q) / (L_a * k))
    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]
    return RGB

# Load HDR image
# hdr_file = "images/images_LeMeur/269/269_HDR2.hdr"
# hdr_img = cv.imread(hdr_file,cv.IMREAD_ANYDEPTH)

# img_fn = ["images/img3.JPG", "images/img2.JPG", "images/img1.JPG", "images/img0.JPG"]
# img_fn = ["images/HDR_src_perso/DSC_0010.JPG", "images/HDR_src_perso/DSC_0009.JPG", "images/HDR_src_perso/DSC_0008.JPG", "images/HDR_src_perso/DSC_0007.JPG"]
img_fn = ["images/images_LeMeur/234/DSC00236.JPG", "images/images_LeMeur/234/DSC00234.JPG", "images/images_LeMeur/234/DSC00235.JPG"]

# Load and align images
img_list = [cv.imread(fn, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH) for fn in img_fn]
alignMBT = cv.createAlignMTB()
alignMBT.process(img_list, img_list)

# Merge images into HDR file
exposure_times = np.array([1.0/60.0, 1.0/320.0, 1.0/2500.0], dtype=np.float32)
merge_debevec = cv.createMergeDebevec()
hdr_img = merge_debevec.process(img_list, times=exposure_times.copy())
cv.imwrite("images/res_hdr.hdr", hdr_img)
cv.imwrite("images/res_hdr.exr", hdr_img)

### TONEMAPS ###

# Drago tonemap
# Params : gamma, saturation, bias
tonemapDrago = cv.createTonemapDrago(gamma=0) #gamma=2.2
ldrDrago = tonemapDrago.process(hdr_img)
ldrDrago_8bit = np.clip(ldrDrago*255, 0, 255).astype('uint8')
cv.imwrite("images/res_drago.jpg", ldrDrago_8bit)
cv.imshow("Drago", ldrDrago_8bit)


# Durand tonemap
#Params : gamma, contrast, saturation, sigma_space, sigma_color
tonemapDurand = cv.createTonemapDurand(gamma=0, contrast=0 ,saturation=0) # gamma=2.2, contrast=5 ,saturation=3
ldrDurand = tonemapDurand.process(hdr_img)
ldrDurand_8bit = np.clip(ldrDurand*255, 0, 255).astype('uint8')
cv.imwrite("images/res_durand.jpg", ldrDurand_8bit)
cv.imshow("Durand", ldrDurand_8bit)

# Mantiuk tonemap
# Params : gamma, scale, saturation
tonemapMantiuk = cv.createTonemapMantiuk(0 ,0, 0) #2.2,0.85, 1.2
ldrMantiuk = tonemapMantiuk.process(hdr_img)
ldrMantiuk_8bit = np.clip(ldrMantiuk*255, 0, 255).astype('uint8')
cv.imwrite("images/res_mantiuk.jpg", ldrMantiuk_8bit)
cv.imshow("Mantiuk", ldrMantiuk_8bit)

# Reinhard tonemap
# Params : gamma, intensity, light_adapt, color_adapt
tonemapReinhard = cv.createTonemapReinhard(0, 0, 0, 0) #1.5, 0, 0, 0
ldrReinhard = tonemapReinhard.process(hdr_img)
ldrReinhard_8bit = np.clip(ldrReinhard*255, 0, 255).astype('uint8')
cv.imwrite("images/res_reinhard.jpg", ldrReinhard_8bit)
cv.imshow("Reinhard", ldrReinhard_8bit)

# Gamma operator
ldrGamma = tonemapping_operator_gamma(hdr_img)
ldrGamma_8bit = np.clip(ldrGamma*255, 0, 255).astype('uint8')
# print("MIN :", np.min(ldrGamma_8bit))
# print("MAX :", np.max(ldrGamma_8bit))
# print(ldrGamma_8bit)
cv.imwrite("images/res_gamma.jpg", ldrGamma)
cv.imshow("Gamma", ldrGamma_8bit)

# Logarithmic operator
ldrLoga = tonemapping_operator_logarithmic(hdr_img)
ldrLoga_8bit = np.clip(ldrLoga*255, 0, 255).astype('uint8')
cv.imwrite("images/res_loga.jpg", ldrLoga_8bit)
cv.imshow("Logarithmic", ldrLoga_8bit)

# Exponential operator
ldrExpo = tonemapping_operator_exponential(hdr_img)
ldrExpo_8bit = np.clip(ldrExpo*255, 0, 255).astype('uint8')
cv.imwrite("images/res_expo.jpg", ldrExpo_8bit)
cv.imshow("Exponential", ldrExpo_8bit)

cv.waitKey()
