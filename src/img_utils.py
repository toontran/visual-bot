"""Text recognition modules"""

"""  
Capture screenshots, extract objects utilities.
"""
import os
import cv2
import numpy as np


def extract_object(template, img, method=cv2.TM_SQDIFF_NORMED):
    '''Returns top-left and bottom-right corners of the detected object.'''
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    h, w, _ = template.shape
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    return top_left, bottom_right

def is_similar(box1, box2, return_dice=False):
    '''If image contents in box1 is similar to box2, return True'''
    THRESHOLD = 0.5
    top_left_1, bottom_right_1 = box1
    top_left_2, bottom_right_2 = box2

    # Dice score
    area_box1 = (bottom_right_1[0] - top_left_1[0]) * (bottom_right_1[1] - top_left_1[1])
    area_box2 = (bottom_right_2[0] - top_left_2[0]) * (bottom_right_2[1] - top_left_2[1])
    
    x_intersect = (min(bottom_right_1[0], bottom_right_2[0]) - max(top_left_1[0], top_left_2[0]))
    y_intersect = (min(bottom_right_1[1], bottom_right_2[1]) - max(top_left_1[1], top_left_2[1]))
    area_intersect = 0
    if x_intersect > 0 and y_intersect > 0:
        area_intersect = x_intersect * y_intersect
    
    dice = (2 * area_intersect) / (area_box1 + area_box2)
    
    if dice > THRESHOLD:
        if return_dice:
            return True, dice
        else:
            return True
        
    if return_dice:
        return False, dice
    else:
        return False

def filter_bounding_boxes(chosen):
    """Helper function for extract_all_objects (changing template size). 
    Deletes overlapping bounding boxes."""
    filtered = []
    while len(chosen) > 0:
        child_of_first_elem = chosen[0]
        keep_indices = []
        for i in range(1, len(chosen)):
            if is_similar(chosen[i], child_of_first_elem):
                pass
            else:
                # Unrelated
                keep_indices.append(i)
        filtered.append(child_of_first_elem)
        chosen = [chosen[i] for i in keep_indices]
    return filtered

def extract_edges(img):    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.max(img_gray) <= 1:
        img_gray = np.uint8(img_gray * 255)
    img_edges = cv2.Canny(img_gray,100,200)
    return img_edges

def extract_all_objects_(template, img, method=cv2.TM_SQDIFF_NORMED, 
                        mean_threshold=None, range_threshold=0.2):
    """Find multiple template(s) in img."""
    if mean_threshold == None:
        if method in [cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF]:
            mean_threshold = 0.2
        else:
            mean_threshold = 0.8
    
    # Using each color channel
    h, w, _ = template.shape
    template = template[...,:3] # RGB only
    img = img[...,:3]
    res_array = np.expand_dims(cv2.matchTemplate(img[:,:,0], template[:,:,0], method), 0)
    for channel in range(1,3):
        res = cv2.matchTemplate(img[:,:,channel], template[:,:,channel], method)
        res_array = np.concatenate([res_array, np.expand_dims(res,0)], axis=0)
    
    # Using edges
    template_edges = extract_edges(template)
    img_edges = extract_edges(img)
    res = cv2.matchTemplate(img_edges, template_edges, method)
    res_array = np.concatenate([res_array, np.expand_dims(res,0)], axis=0)
        
    # Combine
    res_mean = 0.3*res_array[0] + 0.3*res_array[1] + 0.3*res_array[2] + 0.1*res_array[3]
    res_range = mean_threshold - np.min(res_array[:4,:,:], 0)
    
    if method in [cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF]:
        loc = np.where( (res_mean <= mean_threshold) & (res_range <= range_threshold) )
    else:
        loc = np.where( (res_mean >= mean_threshold) & (res_range <= range_threshold) )

    bounding_boxes = [] # Each element contains top left and bottom right of a box
    for pt in zip(*loc[::-1]):
        top_left = pt
        bottom_right = (pt[0] + w, pt[1] + h)
        bounding_boxes.append([top_left, bottom_right])
        
    return bounding_boxes, res_array
            
def extract_all_objects(template, img, method=cv2.TM_SQDIFF_NORMED, 
                        mean_threshold=None, range_threshold=0.2):
    """Extends extract_all_objects_ and allow changing size of template.
    Is 15 times slower than extract_all_objects_"""
    # Function need more work
    reses = []
    max_res = 0
    best_i = None
    chosen_boxes = []
    for i in range(1,15):
        h, w, _ = template.shape
        template_resized = cv2.resize(template, dsize=(w//i, h//i), interpolation=cv2.INTER_LINEAR)
        bounding_boxes, res = extract_all_objects_(template_resized, img, method=cv2.TM_CCOEFF_NORMED, 
                                                  mean_threshold=0.4,
                                                  range_threshold=0.2)
        chosen_boxes.extend(bounding_boxes)
        reses.append(res)
    chosen_boxes = filter_bounding_boxes(chosen_boxes)
    return chosen_boxes
    
