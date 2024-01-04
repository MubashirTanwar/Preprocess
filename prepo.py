import sys
import numpy as np
import cv2
import os


def get_string(img_path):
    
    img = cv2.imread(img_path)

    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split()[0]

    output_path = os.path.join('output_path', "ocr")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rgb_planes = cv2.split(img)
    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)
    

    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1) 
    

    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    

    save_path = os.path.join(output_path, file_name + "_filter_" + str('as') + ".png")
    cv2.imwrite(save_path, img)
    
    
    return save_path

if len(sys.argv) != 2:
    sys.exit("Usage: python ocr.py filename")

image_file = sys.argv[1]

get_string(image_file)
