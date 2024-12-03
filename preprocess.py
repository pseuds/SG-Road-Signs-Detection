import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def resize_image(image, size=(1280,1280), keep_aspect_ratio=True):
    if keep_aspect_ratio:
        h,w,_ = image.shape
        resized_img = cv2.resize(image, (size[1], round(h/w*size[0])))
        rh,rw,_ = resized_img.shape
        border_size = (rw-rh)//2
        resized_img = cv2.copyMakeBorder(resized_img,border_size,border_size,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
    else: 
        resized_img = cv2.resize(image, size)
    return resized_img

def sharpen_image(image, kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])):
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def incr_saturation(img, sat_adj=1.1):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    (h,s,v) = cv2.split(img_hsv)
    s = s*sat_adj
    s = np.clip(s,0,255)
    img_hsv = [h,s,v]
    img_hsv = cv2.merge([h,s,v])
    out = cv2.cvtColor(img_hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return out

def preprocess_img(img):
    out = resize_image(img, size=(800,800))
    out = sharpen_image(out)
    out = incr_saturation(out, sat_adj=2)
    return out

# img_folder = "C:/D drive/2024_T7/CV proj/test_uncropped"
# out_folder = "C:/D drive/2024_T7/CV proj/uncropped_PP"
# dir_ls= os.listdir(img_folder)

# for item_name in dir_ls:
#     if '.png' in item_name:
#         path = img_folder+"/"+item_name
#         img = cv2.imread(path)
#         pp_img = preprocess_img(img)
#         out_path = out_folder+"/"+item_name
#         cv2.imwrite(out_path, pp_img)
#         print(f"Saved as {out_path}.")