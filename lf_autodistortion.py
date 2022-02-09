# Find distortion parameter a, b, c to all *.tiff images in directory
#
# Keys:
#   Line-Selecting:
#     q: done with the image
#     s: set the line starting from last mouse click
#     r: remove the line starting from last mouse click
#     1-9: select line number (only 1-2 in this task)
#   Validation:
#     o: "okay" - accept result for this image
#     q: quit - cancel the program
#     others: redo the last image



import numpy as np
import cv2
from scipy import stats
from scipy import optimize
import glob
import dist
import time
import subprocess
# from matplotlib import pyplot as plt


def str_to_float(string):
    try:
        return float(string)
    except ValueError:
        num, denom = string.split('/')
        return float(num) / float(denom)


def find_lines(img: np.ndarray, num_lines=2): # deprecated
    mouse_x = -1
    mouse_y = -1

    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_x
        nonlocal mouse_y
        # print('MOUSE')
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_x = x
            mouse_y = y
            print(x, ' ', y)

    num_lines_done = 0
    sizex = img.shape[1]
    sizey = img.shape[0]
    lines_y = np.zeros((num_lines, sizex))

    while num_lines_done < num_lines:
        print('Line ', num_lines_done)
        mouse_x = -1
        mouse_y = -1

        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=11)
        sobely = cv2.Sobel(sobely, cv2.CV_64F, 0, 1, ksize=11)
        sobely = sobely[:, :, 0] + sobely[:, :, 1] + sobely[:, :, 2]
        sobely[sobely < 0] = 0
        sobely = sobely / sobely.max()

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)
        cv2.resizeWindow('image', 1800, 900)
        cv2.setMouseCallback('image', mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print('Point: ', mouse_x, ', ', mouse_y)

        if mouse_y >= 0 and mouse_x >= 0:
            y = mouse_y
            lines_img = np.zeros(sobely.shape)
            for x in range(sizex):
                lines_y[num_lines_done, x] = y
                lines_img[y, x] = 1.0
                yplus = y + 1
                yminus = y - 1
                if yplus >= sizey:
                    yplus = sizey - 1
                if yminus < 0:
                    yminus = 0
                if sobely[yplus, x] > sobely[y, x]:
                    y = yplus
                if sobely[yminus, x] > sobely[y, x]:
                    y = yminus

            pimg = np.zeros(img.shape)
            pimg[:, :, 0] = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / (3 * 255)
            pimg[:, :, 1] = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / (3 * 255)
            pimg[:, :, 2] = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / (3 * 255)
            pimg[:, :, 2] = pimg[:, :, 2] + lines_img

            cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
            cv2.imshow('image2', pimg)
            cv2.resizeWindow('image2', 2000, 1000)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key & 0xFF == ord('q'):
                lines_y[0, 0] = -1
                return lines_y
            elif key & 0xFF != ord('r'):
                num_lines_done = num_lines_done + 1

    return lines_y

def find_lines_new(img: np.ndarray, num_lines=2):
    mouse_x = -1
    mouse_y = -1

    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_x
        nonlocal mouse_y
        # print('MOUSE')
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_x = x
            mouse_y = y
            # print(x, ' ', y)

    line_active = 0
    sizex = img.shape[1]
    sizey = img.shape[0]
    lines_y = np.zeros((num_lines, sizex), dtype=int) - 1
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 2, ksize=11)
    # sobely = cv2.Sobel(sobely, cv2.CV_64F, 0, 1, ksize=11)
    sobely = sobely[:, :, 0] + sobely[:, :, 1] + sobely[:, :, 2]
    sobely[sobely < 0] = -sobely[sobely < 0]
    sobely = sobely / sobely.max()

    lines_img = np.zeros(img.shape)

    lines_img = img.copy()
    cv2.namedWindow('DrawLines', cv2.WINDOW_NORMAL)
    cv2.imshow('DrawLines', lines_img)
    cv2.resizeWindow('DrawLines', 2000, 1000)
    cv2.setMouseCallback('DrawLines', mouse_callback)

    active = True
    while active:
        lines_img = img.copy() # can be improved in speed!
        for line in range(num_lines):
            for x in range(sizex):
                if lines_y[line,x] >= 0:
                    lines_img[lines_y[line,x],x,0] = 0.0
                    lines_img[lines_y[line, x], x, 1] = 0.0
                    lines_img[lines_y[line, x], x, 2] = 255
        cv2.imshow('DrawLines', lines_img)
        key = cv2.waitKey(100)
        if key & 0xFF == ord('q'):
            active = False
        elif ord('1') <= key & 0xFF <= ord('9'):
            if (key & 0xFF) - ord('1') < num_lines:
                line_active = (key & 0xFF) - ord('1')
        elif key & 0xFF == ord('r'):
            if mouse_x >= 0:
                for x in range(mouse_x,sizex,1):
                    lines_y[line_active,x] = -1
        elif key & 0xFF == ord('s'):
            if mouse_x >= 0:
                y = mouse_y
                for x in range(mouse_x,sizex,1):
                    lines_y[line_active, x] = y
                    yplus = y + 1
                    yminus = y - 1
                    if yplus >= sizey:
                        yplus = sizey - 1
                    if yminus < 0:
                        yminus = 0
                    if sobely[yplus, x] > sobely[y, x]:
                        y = yplus
                    if sobely[yminus, x] > sobely[y, x]:
                        y = yminus
                lines_y[line_active,sizex-1] = y
        # print('Line_active: ', line_active + 1)

    return lines_y


def min_dist(vals): # deprecated
    global xdi
    global ydi
    global width
    global height
    a = vals[0]
    b = vals[1]
    c = vals[2]
    print('Vals:', a, b, c)
    yui = np.zeros([ydi.shape[1], ydi.shape[0]])
    xui = np.zeros([ydi.shape[1], ydi.shape[0]])
    for v0 in range(ydi.shape[0]):
        for v1 in range(ydi.shape[1]):
            xu, yu = dist.undistort(xdi[v1],ydi[v0,v1],width,height,a,b,c)
            xui[v1,v0] = xu
            yui[v1,v0] = yu
    errorsum = 0.0
    for linenum in range(ydi.shape[0]):
        la, lb, r_value, p_value, std_err = stats.linregress(xui[:,linenum], yui[:,linenum])
        errorsum = errorsum + std_err
    print('Results: ', 1000*errorsum)
    return 1000 * errorsum


def min_dist_new(vals):
    global points
    global width
    global height
    a = vals[0]
    b = vals[1]
    c = vals[2]
    # print('Vals:', a, b, c)
    errorsum = 0.0
    for linenum in range(2):
        a_points = np.array(points[linenum]).transpose()
        xdi = a_points[0,:]
        ydi = a_points[1,:]
        xui = np.zeros(xdi.shape)
        yui = np.zeros(ydi.shape)
        xu_list = []
        yu_list = []
        xui, yui = dist.undistort_vec(xdi,ydi,width,height,a,b,c)
        for pointnum in range(ydi.shape[0]):
            # xui[pointnum], yui[pointnum] = dist.undistort(xdi[pointnum], ydi[pointnum], width, height, a, b, c) # replaced by vectorized version
            if not (np.isnan(xui[pointnum]) or np.isnan(yui[pointnum])):
                xu_list.append(xui[pointnum])
                yu_list.append(yui[pointnum])
        np.delete(xui,np.where(np.isnan(xui)))
        np.delete(yui,np.where(np.isnan(yui)))
        # for pointnum in range(ydi.shape[0]):
        #     xu, yu = dist.undistort(xdi[pointnum],ydi[pointnum],width,height,a,b,c)
        #     if not(np.isnan(xu) or np.isnan(yu)):
        #         xu_list.append(xu)
        #         yu_list.append(yu)
        # print('len', linenum, ': ', len(xu_list)) # Debug to show if enough points are translated for error estimation - programm fails if number gets to low
        xui = np.array(xu_list)
        yui = np.array(yu_list)
        la, lb, r_value, p_value, std_err = stats.linregress(xui, yui)
        errorsum = errorsum + std_err*std_err
    # print('Results: ', 1000*errorsum)
    return 1000 * errorsum


file_list = glob.glob('*.tiff')
file_list.sort()

file_index = 0 # Number of processed image
active = True

file_results = open('results.txt', 'w')

while active:
    image = cv2.imread(file_list[file_index], cv2.IMREAD_COLOR)

    # get exif data:
    exif_focallength = subprocess.check_output(["exiv2", "-K", "Exif.Photo.FocalLength", "-P", "v", file_list[file_index]])
    focallength = str_to_float(str(exif_focallength)[2:-3])
    
    # TODO: what happens if no exiv2 available? 

    l_y = find_lines_new(image, 2)
    
   
    width = image.shape[1]
    height = image.shape[0]

    cv2.destroyAllWindows()
    
    # Select a subset of points on lines for regression

    points = []
    for line in range(2):
        points.append([])
        for x in range(20, width - 20, 50):
            if l_y[line, x] >= 0:
                points[line].append([x, l_y[line, x]])

    initial_guess = np.array([0.01, 0.01, 0.01])
    #initial_guess = np.array([0.0266783285002862, -0.0864150044158616, 0.0509941823684105])

    #res = optimize.minimize(min_dist, initial_guess)

    #res = optimize.least_squares(min_dist_new, initial_guess, method='lm')

    #res = optimize.basinhopping(min_dist_new, [(-1,1),(-1,1),])
    
    # Two step approach seems to work quit good

    bounds = [(-1, 1), (-1, 1), (-1, 1)]

    t1 = time.time()
    res = optimize.dual_annealing(min_dist_new, bounds)
    t2 = time.time()
    print('Optimization time 1: ', t2-t1)

    t1 = time.time()
    res = optimize.least_squares(min_dist_new, res.x)
    t2 = time.time()
    print('Optimization time 2: ', t2 - t1)

    a = res.x[0]
    b = res.x[1]
    c = res.x[2]
    print(file_list[file_index], ': a = ', a, ', b = ', b, ', c = ', c)
    
    # Show results, visualize selected lines with straight red lines
    
    img_lines = dist.image_undistort(image, a, b, c)
    for linenum in range(2):
        a_points = np.array(points[linenum]).transpose()
        xdi = a_points[0, :]
        ydi = a_points[1, :]
        xui = np.zeros(xdi.shape)
        yui = np.zeros(ydi.shape)
        for pointnum in range(ydi.shape[0]):
            xu, yu = dist.undistort(xdi[pointnum], ydi[pointnum], width, height, a, b, c)
            xui[pointnum] = xu
            yui[pointnum] = yu
        la, lb, r_value, p_value, std_err = stats.linregress(xui, yui)
        for x in range(width):
            y = int(round(la * x + lb))
            if 0 <= y < height:
                img_lines[y, x, 0] = 0
                img_lines[y, x, 1] = 0
                img_lines[y, x, 2] = 255

    cv2.namedWindow('Check', cv2.WINDOW_NORMAL)
    cv2.imshow('Check', img_lines / 255)
    cv2.resizeWindow('Check', 1800, 900)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key & 0xFF == ord('q'):
        active =  False
    elif key & 0xFF == ord('o'):
        result_str = 'distortion(' + str(focallength) + 'mm) = ' + str(a) + ', ' + str(b) + ', ' + str(c) + '\n'
        file_results.write(result_str)
        file_index += 1
        if file_index >= len(file_list):
            active = False

file_results.close()















