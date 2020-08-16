import cv2
import numpy as np
import copy
from tkinter import filedialog
from pyautocad import Autocad, APoint
import os


color_white = np.array([255,255,255])
color_blue = np.array([255,130,130])
color_gray = np.array([ 80, 80, 80])

#def getRepresentativeColor(color_list):


def sign(p1,p2,p3):
    return (((p1[0] - p3[0]) * (p2[1] - p3[1])) - ((p2[0] - p3[0]) * (p1[1] - p3[1])))

def PointInTriangle(pt, v1, v2, v3):
    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = np.logical_or(np.logical_or((d1 < 0), (d2 < 0)), (d3 < 0))
    has_pos = np.logical_or(np.logical_or((d1 > 0), (d2 > 0)), (d3 > 0))    

    return np.logical_not( np.logical_and(has_neg, has_pos) )

def getColorData(img, p1, p2, p3):
    start_x = min(min(p1[0], p2[0]), p3[0])
    end_x = max(max(p1[0], p2[0]), p3[0])
    start_y = min(min(p1[1], p2[1]), p3[1])
    end_y = max(max(p1[1], p2[1]), p3[1])

    color_list = []
    pos_list = []
    for y in range(int(start_y), int(end_y)):
        for x in range(int(start_x), int(end_x)):
            point = np.array([x,y])
            if PointInTriangle(point, p1, p2, p3):
                color_list.append(img[y,x])
                pos_list.append(point)
    return color_list, pos_list

def calculateColor(color_list):
    return np.average(color_list, axis=0).tolist()
    

def getSampleValue(img):
    #cv2.imshow(img)
    
    debug_img = copy.copy(img)

    center_y = img.shape[0] / 2
    center_x = img.shape[1] / 2

    row_size_y = img.shape[0]
    col_size_x = img.shape[1]
    
    # Top
    top_p1 = np.array([0,0])
    top_p2 = np.array([center_x,center_y])
    top_p3 = np.array([col_size_x,0])
    color_list, pos_list = getColorData(img, top_p1, top_p2, top_p3)
    color_top = calculateColor(color_list)
    
    # left
    left_p1 = np.array([0,0])
    left_p2 = np.array([center_x,center_y])
    left_p3 = np.array([0,row_size_y])
    color_list, pos_list = getColorData(img, left_p1, left_p2, left_p3)
    color_left = calculateColor(color_list)


    # right
    right_p1 = np.array([col_size_x,row_size_y])
    right_p2 = np.array([center_x,center_y])
    right_p3 = np.array([col_size_x,0])
    color_list, pos_list = getColorData(img, right_p1, right_p2, right_p3)
    color_right = calculateColor(color_list)

    # bot
    bot_p1 = np.array([col_size_x,row_size_y])
    bot_p2 = np.array([center_x,center_y])
    bot_p3 = np.array([0,row_size_y])
    color_list, pos_list = getColorData(img, bot_p1, bot_p2, bot_p3)
    color_bot = calculateColor(color_list)

    target_size = 200
    debug_img = cv2.resize(debug_img, dsize=(200, 200), interpolation=cv2.INTER_NEAREST)
    cv2.line( debug_img, (0,0), (target_size, target_size), 1)
    cv2.line( debug_img, (target_size,0), (0, target_size), 1)

    ### Draw Color
    debug_output_img = np.zeros(debug_img.shape, np.uint8)
    pts = np.array([(0,0), (target_size,0), (target_size/2, target_size/2)], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(debug_output_img, [pts], color_top) 
    
    pts = np.array([(0,0), (0,target_size), (target_size/2, target_size/2)], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(debug_output_img, [pts], color_left) 
    
    pts = np.array([(target_size,target_size), (target_size,0), (target_size/2, target_size/2)], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(debug_output_img, [pts], color_right) 
    
    pts = np.array([(target_size,target_size), (0,target_size), (target_size/2, target_size/2)], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(debug_output_img, [pts], color_bot) 

    # cv2.imshow("debug_img", debug_img)
    # cv2.imshow("debug_img_ouput", debug_output_img)
    # cv2.waitKey(0)

    return color_top, color_left, color_right, color_bot

if __name__=="__main__":
    row_num = 36
    col_num = row_num


    filename = filedialog.askopenfilename(initialdir="/",
                                            title="Select file",
                                            filetypes=(("jpeg","*.jpg"),("png","*.png"),("all files","*.*")))
    print(filename)
    img = cv2.imread(filename)

    if img.shape[0] != img.shape[1]:
        print("Shape is wrong")
        exit()

    width = img.shape[0] / row_num

    debug_output_img = copy.copy(img)

    row_center = img.shape[0] / 2
    col_center = img.shape[1] / 2

    print(row_center, col_center)
    cv2.circle(img, (int(col_center), int(row_center)), 3, (255,255,255), -1)

    start_pos_list = np.indices((row_num, col_num)) * width
    start_pos_list[0] = start_pos_list[0] + row_center - (row_num * width / 2.0)
    start_pos_list[1] = start_pos_list[1] + col_center - (col_num * width / 2.0)

    end_pos_list = np.indices((row_num, col_num)) * width
    end_pos_list[0] = end_pos_list[0] + row_center - ((row_num-2) * width / 2.0)
    end_pos_list[1] = end_pos_list[1] + col_center - ((col_num-2) * width / 2.0)

    center_pos_list = np.indices((row_num, col_num)) * width
    center_pos_list[0] = center_pos_list[0] + row_center - ((row_num-1) * width / 2.0)
    center_pos_list[1] = center_pos_list[1] + col_center - ((col_num-1) * width / 2.0)

    output_img = np.zeros((row_num,col_num,3), np.uint8)

    for row_idx, (y1_list, x1_list, yc_list, xc_list, y2_list, x2_list) in enumerate(zip(start_pos_list[0], start_pos_list[1], center_pos_list[0], center_pos_list[1], end_pos_list[0], end_pos_list[1])):
        for col_idx, (y1, x1, yc, xc, y2, x2) in enumerate(zip(y1_list, x1_list, yc_list, xc_list, y2_list, x2_list)):
            
            
            #print(y,x)
            # cv2.circle(img, (int(x1),int(y1)), 1, (255,0,0), -1)
            # cv2.circle(img, (int(x2),int(y2)), 1, (255,255,0), -1)
            # cv2.circle(img, (int(xc),int(yc)), 1, (0,0,255), -1)
            # continue

            #cv2.rectangle( img, (int(x2),int(y2)), (int(x1),int(y1)) , (255,0,0) , 1)
            #continue
            
            color_top, color_left, color_right, color_bot = getSampleValue(img[int(y1):int(y2),int(x1):int(x2)])

            pts = np.array([(x1,y1), (x2,y1), (xc,yc)], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(debug_output_img, [pts], color_top) 
            
            pts = np.array([(x1,y1), (x1,y2), (xc,yc)], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(debug_output_img, [pts], color_left)

            pts = np.array([(x2,y2), (x2,y1), (xc,yc)], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(debug_output_img, [pts], color_right)
            
            pts = np.array([(x2,y2), (x1,y2), (xc,yc)], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(debug_output_img, [pts], color_bot)



    # for y in range(row_num):
    #     for x in range(col_num):
    #         print(x, y)
    # output_img = cv2.resize(output_img, dsize=(1000, 1000), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow("test2", output_img)

    debug_output_img = cv2.resize(debug_output_img, dsize=(1000, 1000), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("test2", debug_output_img)

    target_path = os.path.splitext(filename)[0] + "_output.png"
    cv2.imwrite(target_path, debug_output_img)

    cv2.waitKey(0)

