import cv2
import numpy as np
import math
import sys

lim_delta = 1
delta = 0

# Identifying the edges:
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 200,400)
    return canny

def steering_correction(central_line):
    global delta
    temp_delta = 90-abs(math.atan(central_line[0])*(180/np.pi))
    
    # Checking wether the correction angle delta is bigger than lim_delta_degrees
    if abs(delta-temp_delta) > lim_delta:
        delta = temp_delta
    
    if central_line[0] < 0:
        #print("Turn", delta, "degrees left" )
        text = "Turn " + str(delta) + " degrees left"
    else:
        #print("Turn", delta, "degrees right")
        text = "Turn " + str(delta) + " degrees right"
        
    return text
        
def show_img(image, central_line):
    cv2.imshow("Frame", image)

def region_of_interest(edges):
    height, width = edges.shape
    poly = np.array([[(height,height), (0,height*(1/2)), (width, height*(1/2)), (width, height)]], np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, poly, 255)
    cropped_edges = cv2.bitwise_and(mask, edges)
    return cropped_edges

def find_lines(cropped):
    rho = 1
    angle = np.pi/180
    min_threshold = 50
    line_segments = cv2.HoughLinesP(
        cropped,
        rho,
        angle,
        min_threshold,
        np.array([]),
        minLineLength=8,
        maxLineGap=4
    )
    
    return line_segments

def make_coordinates(image, lines):
    if lines.shape[0] == 0:
        return
    slope, intercept = lines
    y1 = image.shape[0]
    y2 = int(y1*(1/2))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return [x1,y1,x2,y2] 

def average_slope_intercept(image, lines):
    
    left_lines = []
    right_lines = []
    
    if lines is None:
        print("The robot cannot recognize anything!")
        return None
    
    for i in range(lines.shape[0]):
        x1,y1,x2,y2 = lines[i][0]
        if x1 != x2 and abs(y1-y2)>10: 
            fit = np.polyfit((x1,x2),(y1,y2), 1)
            if fit[0]>0:
                right_lines.append(fit)
            else:
                left_lines.append(fit)
            
    if len(left_lines)>0:       
        left_average = np.average(left_lines, axis = 0)   
    else:
        left_average = np.array([])
        
    if len(right_lines)>0:
        right_average = np.average(right_lines, axis = 0)
    else:
        right_average = np.array([])

    if len(right_average) != 0 and len(left_average) != 0:
        return [make_coordinates(image, left_average), make_coordinates(image, right_average)]
    
    return None

def make_central_line(image, lines):
    if lines[0] is None:
        middle_line = lines[1]
        x1,y1,x2,y2 = middle_line
        fit = np.polyfit((x1,x2),(y1,y2), 1)
        return [make_coordinates(image,fit), fit]
    if lines[1] is None:
        middle_line = lines[0]
        x1,y1,x2,y2 = middle_line
        fit = np.polyfit((x1,x2),(y1,y2), 1)
        return [make_coordinates(image,fit), fit]
    
    middle_line = np.average(lines, axis = 0)
    x1,y1,x2,y2 = middle_line
    fit = np.polyfit((x1,x2),(y1,y2), 1)
    return [make_coordinates(image,fit), fit]

def display_lines(image, lines, central_line):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            x1,y1,x2,y2 = line
            cv2.line(line_image, (x1,y1),(x2,y2),(0,0,255),5)
    x1,y1,x2,y2 = central_line
    cv2.line(line_image, (x1,y1),(x2,y2),(255,0,0),5)
    return line_image

def image_vision(image, countFrame):
    edges = canny(image)
    cropped_edges = region_of_interest(edges)
    lines = average_slope_intercept(image, find_lines(cropped_edges))
    if lines is not None:
        central_line = make_central_line(image, lines)
        line_image = display_lines(image, lines, central_line[0])
        combo_image = cv2.addWeighted(line_image, 0.8, image, 1, 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combo_image, steering_correction(central_line[1]), (50,50), font, .5, (255,255,255))
        show_img(combo_image, central_line[1])

def main():
    path_video_capture = 0 if sys.argv[1] == '0' else sys.argv[1]
    video = cv2.VideoCapture(path_video_capture)
    countFrame = 0
    while True:
        print(countFrame)
        ret, frame = video.read()
        if not ret:
            print("Problem while trying to read the video file.")
            break
        
        # Putting the image in the resolution of Arduino's camera:
        frame = cv2.resize(frame, (640,480))
        
        image_vision(frame, countFrame)
        countFrame += 1
        if cv2.waitKey(10) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    main()