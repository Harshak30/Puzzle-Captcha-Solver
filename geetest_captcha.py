# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:10:00 2022
@author: Harsha

Requirements:
    Python 3.7
    Chrome Version 97.0.4692.71
    cv2 version 4.2.0
    Selenium version 3.141.0
"""

from selenium import webdriver
import base64
import cv2
import os
import time
import numpy as np
from selenium.webdriver.common.action_chains import ActionChains
from matplotlib import pyplot as plt
PIXELS_EXTENSION = 0

#Preprocessing the image and finding the missing puzzle piece position
class PuzleSolver:
    def __init__(self, piece_path, background_path):
        self.piece_path = piece_path
        self.background_path = background_path

    def get_position(self):
        template, x_inf, y_sup, y_inf = self.__piece_preprocessing()
        background = self.__background_preprocessing(y_sup, y_inf)
        res = cv2.matchTemplate(background, template, cv2.TM_CCOEFF_NORMED)
        w, h = template.shape[::-1]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(background,top_left, bottom_right, 255, 2)
        plt.subplot(122),plt.imshow(background,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.show()
        origin = x_inf
        end = top_left[0] + PIXELS_EXTENSION
        return end-origin

    def __background_preprocessing(self, y_sup, y_inf):
        background = self.__sobel_operator(self.background_path)
        background = background[y_sup:y_inf, :]
        background = self.__extend_background_boundary(background)
        background = self.__img_to_grayscale(background)
        return background

    def __piece_preprocessing(self):
        img = self.__sobel_operator(self.piece_path)
        x, w, y, h = self.__crop_piece(img)
        template = img[y:h, x:w]

        template = self.__extend_template_boundary(template)
        template = self.__img_to_grayscale(template)

        return template, x, y, h

    def __crop_piece(self, img):
        white_rows = []
        white_columns = []
        r, c = img.shape
        
        for row in range(r):
            for x in img[row, :]:
                if x != 0:
                    white_rows.append(row)

        for column in range(c):
            for x in img[:, column]:
                if x != 0:
                    white_columns.append(column)

        x = white_columns[0]
        w = white_columns[-1]
        y = white_rows[0]
        h = white_rows[-1]

        return x, w, y, h

    def __extend_template_boundary(self, template):
        extra_border = np.zeros((template.shape[0], PIXELS_EXTENSION), dtype=int)
        template = np.hstack((extra_border, template, extra_border))

        extra_border = np.zeros((PIXELS_EXTENSION, template.shape[1]), dtype=int)
        template = np.vstack((extra_border, template, extra_border))

        return template

    def __extend_background_boundary(self, background):
        extra_border = np.zeros((PIXELS_EXTENSION, background.shape[1]), dtype=int)
        return np.vstack((extra_border, background, extra_border))

    def __sobel_operator(self, img_path):
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #to detect edges
        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0) #blending
        return grad

    def __img_to_grayscale(self, img):
        tmp_path = "d:/output/sobel1.png"
        cv2.imwrite(tmp_path, img)
        return cv2.imread(tmp_path, 0)


'''
IMAGE SCRAPING
'''
# create webdriver object
driver = webdriver.Chrome('C:/Users/User/Downloads/chromedriver_win32/chromedriver.exe')
#get website
driver.get("https://2captcha.com/demo/geetest")
#Maximise browser window
driver.maximize_window()
time.sleep(10)
element=driver.find_element_by_class_name("geetest_holder")
element.click()
time.sleep(10)

#Scraping the website to get the canvas and Converting canvas to image format
base64_image = driver.execute_script("return document.querySelector('canvas.geetest_canvas_bg').toDataURL('image/png').substring(21);")
piece = driver.execute_script("return document.querySelector('canvas.geetest_canvas_slice').toDataURL('image/png').substring(21);")
output_image = base64.b64decode(base64_image)
piece_image = base64.b64decode(piece)

# save image
with open("d:/output/background.png", 'wb') as f:
   f.write(output_image)
   
with open("d:/output/piece.png", 'wb') as f:
   f.write(piece_image)
   

'''
Get position of missing piece  
'''    
solver = PuzleSolver("d:/output/piece.png", "d:/output/background.png")
solution= solver.get_position()

print("position",solution)  

#geetest_slider_button
slider = driver.find_element_by_css_selector("div.geetest_slider_button")
move = ActionChains(driver)
time.sleep(2)

ActionChains(driver).click_and_hold(slider).perform()
time.sleep(0.3)
ActionChains(driver).move_by_offset(xoffset=solution-25, yoffset=0).perform()
time.sleep(0.03)
ActionChains(driver).move_by_offset(xoffset=12, yoffset=0).perform()
time.sleep(0.03)
ActionChains(driver).move_by_offset(xoffset=9, yoffset=0).perform()
time.sleep(0.01)
ActionChains(driver).move_by_offset(xoffset=3, yoffset=0).perform()
time.sleep(0.01)
ActionChains(driver).move_by_offset(xoffset=1, yoffset=0).perform()
time.sleep(0.05)
ActionChains(driver).release().perform()


'''
Adding Timer between Each Move

val=1
for i in range(1,solution):
    time.sleep(0.1)
    ActionChains(driver).click_and_hold(slider).move_by_offset(xoffset=val, yoffset=0).perform()
ActionChains(driver).release().perform()
'''

folder_path = (r'D:\output')
test = os.listdir(folder_path)
for images in test:
    if images.endswith(".png"):
        os.remove(os.path.join(folder_path, images))