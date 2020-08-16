import cv2
import numpy as np
import copy
import os
from tkinter import filedialog

def drawImage(display_img, rect_info, selected_idx):
    line_color = (250,250,100)
    cv2.rectangle( display_img, (int(rect_info[0,0]), int(rect_info[0,1])),(int(rect_info[1,0]), int(rect_info[1,1])), line_color, 1 )
    
    if selected_idx == 0:
        cv2.circle( display_img, (int(rect_info[0,0]), int(rect_info[0,1])),5 , (0,0,230), -1 )
    elif selected_idx == 1:
        cv2.circle( display_img, (int(rect_info[1,0]), int(rect_info[1,1])),5 , (0,0,230), -1 )

    # center_x = rect_info[0,0] + (rect_info[1,0] - rect_info[0,0]) / 2
    # center_y = rect_info[0,1] + (rect_info[1,1] - rect_info[0,1]) / 2
    # cv2.circle( display_img, (int(center_x), int(center_y)), 3, (255,255,0), -1)

    cv2.line( display_img, (int(rect_info[0,0]), int(rect_info[0,1])),(int(rect_info[1,0]), int(rect_info[1,1])), line_color, 1 )
    cv2.line( display_img, (int(rect_info[0,0]), int(rect_info[1,1])),(int(rect_info[1,0]), int(rect_info[0,1])), line_color, 1 )

    return display_img

def loadJson(path):
    json_data = {}
    with open(path, "r") as json_file:
        json_data = json.load(json_file)

    return json_data

def saveJson(path, region_info):
    data = {}
    data["region_info"] = region_info.tolist()
    data["resol_info"] = resol_info.tolist()

    with open(path, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    
    return region_info, resol_info

if __name__=="__main__":
    filename = filedialog.askopenfilename(initialdir="/",
                                            title="Select file",
                                            filetypes=(("jpeg","*.jpg"),("png","*.png"),("all files","*.*")))
    print(filename)

    img = cv2.imread(filename)

    #cv2.imshow("test",img)
    #cv2.waitKey(0)

    selected_index = 0

    start_x = img.shape[1] / 4.0
    start_y = img.shape[0] / 4.0
    end_x = img.shape[1] * 3.0 / 4.0
    end_y = img.shape[0] * 3.0 / 4.0

    target_region = np.array([[start_x, start_y],[end_x, end_y]])
    resol_info = [100,100]

    while True:
        display_img = copy.copy(img)
        display_img = drawImage(display_img, target_region, selected_index)
        cv2.imshow("display_img", display_img)
        ch = cv2.waitKey(0)
        if ch == ord('q'):
            break
        elif ch == ord('1'):
            selected_index = 0
        elif ch == ord('2'):
            selected_index = 1
        elif ch == ord('w'):
            target_region[selected_index, 1] -= 1
        elif ch == ord('a'):
            target_region[selected_index, 0] -= 1
        elif ch == ord('s'):
            target_region[selected_index, 1] += 1
        elif ch == ord('d'):
            target_region[selected_index, 0] += 1
        elif ch == ord('y'):
            target_path = os.path.splitext(filename)[0] + "_cropped.png"
            print(target_path)
            output_img = img[int(target_region[0,1]):int(target_region[1,1]), int(target_region[0,0]):int(target_region[1,0]), :]
            output_img = cv2.resize(output_img, dsize=(1000, 1000), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(target_path, output_img)
            break

        #print(ch)