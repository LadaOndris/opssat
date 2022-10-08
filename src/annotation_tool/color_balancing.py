import cv2
import numpy as np


def simplest_cb(img, percent=1):
    """
    From: JackDesBwa, Sep 12, 2019
    https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
    """
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0, 256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)


if __name__ == '__main__':
    cv2.namedWindow("before", cv2.WINDOW_NORMAL)
    cv2.namedWindow("after", cv2.WINDOW_NORMAL)
    img = cv2.imread('datasets/opssat/raw/006.png')
    out = simplest_cb(img, 1)
    cv2.imshow('before', img)
    cv2.imshow('after', out)

    key = cv2.waitKey(0)
    while key != 27:
        key = cv2.waitKey(0)
