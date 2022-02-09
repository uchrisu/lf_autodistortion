import numpy as np
import numpy.matlib
import time


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





