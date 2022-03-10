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
import numpy.matlib
import cv2
from scipy import stats
from scipy import optimize
import glob
import time
import subprocess
import sys


###### Distortion functions #######

def distort(x, y, width, height, a, b, c):
    r_full = height/2  # np.hypot(width, height)/2
    x_ = (x - width / 2) / r_full
    y_ = (y - height / 2) / r_full
    ru = np.hypot(x_, y_)
    # rd = ru * (a*ru*ru*ru + b*ru*ru + c*ru + 1 - a - b - c)
    factor = (a*ru*ru*ru + b*ru*ru + c*ru + 1 - a - b - c)
    xd_ = factor * x_
    yd_ = factor * y_
    xd = xd_ * r_full + width / 2
    yd = yd_ * r_full + height / 2
    return xd, yd


def undistort(xd, yd, width, height, a, b, c):
    # Newton's method
    EPS = 1e-5
    r_full = height/2
    xd_ = (xd - width/2) / r_full
    yd_ = (yd - height/2) / r_full
    rd = np.hypot(xd_, yd_)

    # initial guess:
    ru = rd
    step = 0
    while 1:
        step = step + 1
        ru2 = ru*ru
        diff = ru * (a*ru*ru2 + b*ru2 + c*ru + 1 - a - b - c) - rd
        if abs(diff) < EPS:
            break
        if step > 10:
            ru = np.nan
            break
        div = 4 * a * ru * ru2 + 3 * b * ru2 + 2 * c * ru + 1 - a - b - c
        ru -= diff/div
    if ru < 1e-5:
        xu_ = xd_/(1-a-b-c)
        yu_ = yd_/(1-a-b-c)
    else:
        xu_ = xd_ * ru/rd
        yu_ = yd_ * ru/rd
    xu = xu_ * r_full + width/2
    yu = yu_ * r_full + height/2
    return xu, yu


def undistort_vec(xd, yd, width, height, a, b, c):
    # vectorized version of undistort
    EPS = 1e-5
    r_full = height/2
    xd_ = (xd - width/2) / r_full
    yd_ = (yd - height/2) / r_full
    rd = np.hypot(xd_, yd_)
    ru = rd
    d = 1 - a - b - c
    for step in range(10):
        ru2 = np.square(ru)
        ru3 = np.multiply(ru2, ru)
        diff = np.multiply(a * ru3 + b * ru2 + c * ru + d, ru) - rd
        div = (4 * a) * ru3 + (3 * b) * ru2 + (2 * c) * ru + d
        ru = ru - np.divide(diff, div)
    ru2 = np.square(ru)
    ru3 = np.multiply(ru2, ru)
    diff = np.multiply(a * ru3 + b * ru2 + c * ru + d, ru) - rd
    ru[np.abs(diff) > EPS] = np.nan
    factor = np.divide(ru, rd)
    xu_ = np.multiply(xd_, factor)
    yu_ = np.multiply(yd_, factor)
    dinv = 1/d
    xu_[ru < EPS] = xd_[ru < EPS] * dinv
    yu_[ru < EPS] = yd_[ru < EPS] * dinv
    xu = xu_ * r_full + width/2
    yu = yu_ * r_full + height/2
    return xu, yu


def image_undistort(img: np.ndarray, a, b, c):
    # return an undistorted version of img, vectorized implementation
    # t1 = time.time()
    width = img.shape[1]
    height = img.shape[0]
    color_num = img.shape[2]
    r_full = height / 2
    xuv = np.arange(width)
    yuv = np.arange(height)
    xuv_ = (xuv - width / 2) / r_full
    yuv_ = (yuv - height / 2) / r_full
    xum_ = np.matlib.repmat(xuv_, height, 1)
    yum_ = np.matlib.repmat(yuv_.reshape(height,1), 1, width)
    ru = np.hypot(xum_, yum_)
    ru2 = np.multiply(ru, ru)
    factor = (a * np.multiply(ru2, ru) + b * ru2 + c * ru + (1 - a - b - c))
    xdm_ = np.multiply(xum_, factor)
    ydm_ = np.multiply(yum_, factor)
    xdm = np.round((xdm_ * r_full) + width/2).astype(np.int32)
    ydm = np.round((ydm_ * r_full) + height/2).astype(np.int32)
    imgc = np.zeros(img.shape)
    # t2 = time.time()
    # print('Time Part1: ', t2 - t1)
    xdm_fast = xdm.copy()
    ydm_fast = ydm.copy()
    xdm_fast[xdm < 0] = 0
    xdm_fast[xdm >= width] = 0
    ydm_fast[ydm < 0] = 0
    ydm_fast[ydm >= height] = 0
    for color in range(color_num):
        imgc[:, :, color] = img[ydm_fast[:, :], xdm_fast[:, :], color]
        # for x in range(width):
        #     imgc[:, x, color] = img[ydm[:, x], xdm[:, x], color]
        #     # for y in range(height):
        #     #
        #     #     if xdm[y,x] < 0 or xdm[y,x] >= width or ydm[y,x] < 0 or ydm[y,x] >= height:
        #     #         imgc[y, x, color] = 0
        #     #     else:
        #     #         imgc[y, x, color] = img[ydm[y,x], xdm[y,x], color]
    imgc[xdm < 0] = 0
    imgc[xdm >= width] = 0
    imgc[ydm < 0] = 0
    imgc[ydm >= height] = 0
    return imgc


#############################


def str_to_float(string):
    try:
        return float(string)
    except ValueError:
        num, denom = string.split('/')
        return float(num) / float(denom)


def norm_cross_corr(v1, v2):
    v1m = np.mean(v1)
    v2m = np.mean(v2)
    s12 = np.sum((v1-v1m)*(v2-v2m))
    s11 = np.sum((v1-v1m)*(v1-v1m))
    s22 = np.sum((v2-v2m)*(v2-v2m))
    return s12/np.sqrt(s11*s22)


def find_lines_sobel(img: np.ndarray, num_lines=2):
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
    sobely = np.abs(sobely[:, :, 0]) + np.abs(sobely[:, :, 1]) + np.abs(sobely[:, :, 2])
    # sobely[sobely < 0] = -sobely[sobely < 0]
    sobely = sobely / sobely.max()

    lines_img = np.zeros(img.shape)

    lines_img = img.copy()
    cv2.namedWindow('DrawLines', cv2.WINDOW_NORMAL)
    cv2.imshow('DrawLines', lines_img)
    cv2.resizeWindow('DrawLines', 2000, 1000)
    cv2.setMouseCallback('DrawLines', mouse_callback)
    cv2.setWindowTitle('DrawLines',
                       'Draw Lines. 1,2: select 1st/2nd line, q: done, s,r: start/remove line from here (last click)')

    active = True
    while active:
        lines_img = img.copy() # can be improved in speed!
        for line in range(num_lines):
            for x in range(sizex):
                if lines_y[line, x] >= 0:
                    y = lines_y[line, x]
                    if x % 4 == 0:
                        lines_img[y - 5:y + 6, x, 0] //= 2
                        lines_img[y - 5:y + 6, x, 1] //= 2
                        lines_img[y - 5:y + 6, x, 2] //= 2
                        lines_img[y - 5:y + 6, x, 1] += 127
                    if x % 2 == 1:
                        lines_img[y, x, 0] = 0.0
                        lines_img[y, x, 1] = 255
                        lines_img[y, x, 2] = 0.0
                    else:
                        lines_img[y, x, 0] = 255
                        lines_img[y, x, 1] = 0.0
                        lines_img[y, x, 2] = 0.0
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


def find_lines_xcorr(img: np.ndarray, num_lines=2):
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

    lines_img = np.zeros(img.shape)

    lines_img = img.copy()
    cv2.namedWindow('DrawLines', cv2.WINDOW_NORMAL)
    cv2.imshow('DrawLines', lines_img)
    cv2.resizeWindow('DrawLines', 2000, 1000)
    cv2.setMouseCallback('DrawLines', mouse_callback)
    cv2.setWindowTitle('DrawLines',
                       'Draw Lines. 1,2: select 1st/2nd line, q: done, s,r: start/remove line from here (last click)')

    active = True
    while active:
        lines_img = img.copy() # can be improved in speed!
        for line in range(num_lines):
            for x in range(sizex):
                if lines_y[line, x] >= 0:
                    y = lines_y[line, x]
                    if x % 4 == 0:
                        lines_img[y - 5:y + 6, x, 0] //= 2
                        lines_img[y - 5:y + 6, x, 1] //= 2
                        lines_img[y - 5:y + 6, x, 2] //= 2
                        lines_img[y - 5:y + 6, x, 1] += 127
                    if x % 2 == 1:
                        lines_img[y, x, 0] = 0.0
                        lines_img[y, x, 1] = 255
                        lines_img[y, x, 2] = 0.0
                    else:
                        lines_img[y, x, 0] = 255
                        lines_img[y, x, 1] = 0.0
                        lines_img[y, x, 2] = 0.0
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
                if y >= sizey - 2:
                    y = sizey - 3
                if y < 2:
                    y = 2
                target = img[y-2:y+3, mouse_x]
                for x in range(mouse_x, sizex, 1):
                    lines_y[line_active, x] = y
                    yplus = y + 1
                    yminus = y - 1
                    if yplus >= sizey - 2:
                        yplus = sizey - 3
                    if yminus < 2:
                        yminus = 2
                    val_minus = norm_cross_corr(target, img[yminus - 2:yminus + 3, x])
                    val_same = norm_cross_corr(target, img[y - 2:y + 3, x])
                    val_plus = norm_cross_corr(target, img[yplus - 2:yplus + 3, x])
                    if val_plus > val_same:
                        y = yplus
                    if val_minus > val_same and val_minus > val_plus:
                        y = yminus
                lines_y[line_active,sizex-1] = y

    return lines_y


def min_dist(vals):  # deprecated
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
            xu, yu = undistort(xdi[v1],ydi[v0,v1],width,height,a,b,c)
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
        xui, yui = undistort_vec(xdi,ydi,width,height,a,b,c)
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


points = []
width = 0
height = 0
xdi = np.array([])
ydi = np.array([])


def main():
    global width
    global height
    global points
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

        l_y = find_lines_xcorr(image, 2)

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

        if len(points[0]) < 10 or len(points[1]) < 10:
            key = input('Error: Not enough Points, retry? (y/n) ')
            if key == 'y' or key == 'Y':
                continue
            sys.exit()

        print('Calculating. Please wait...')

        #initial_guess = np.array([0.01, 0.01, 0.01])

        #res = optimize.minimize(min_dist, initial_guess)

        #res = optimize.least_squares(min_dist_new, initial_guess, method='lm')

        #res = optimize.basinhopping(min_dist_new, [(-1,1),(-1,1),])

        # Two step approach seems to work quite good

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

        img_lines = image_undistort(image, a, b, c)
        for linenum in range(2):
            a_points = np.array(points[linenum]).transpose()
            xdi = a_points[0, :]
            ydi = a_points[1, :]
            xui = np.zeros(xdi.shape)
            yui = np.zeros(ydi.shape)
            for pointnum in range(ydi.shape[0]):
                xu, yu = undistort(xdi[pointnum], ydi[pointnum], width, height, a, b, c)
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
        cv2.setWindowTitle('Check', 'Check Lines. o: line is okay, r: reject results, q: quit/abort program')
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key & 0xFF == ord('q'):
            active =  False
        elif key & 0xFF == ord('o'):
            result_str = 'distortion(' + str(focallength) + 'mm) = ' + str(round(a,7)) + ', ' + str(round(b,7)) + ', ' + str(round(c,7)) + '\n'
            file_results.write(result_str)
            file_index += 1
            if file_index >= len(file_list):
                active = False

    file_results.close()


if __name__ == '__main__':
    main()
