import cv2
import numpy as np

def watershed(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    return img
def grab_cut(img):
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (50, 50, 450, 290)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 1, 0).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    return img

def simple_thresholding(img):
    _ , ret = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return ret
def adaptive_thresholding(img):
    img = cv2.medianBlur(img, 5)
    ret = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return ret

def otsu_report(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    otsu_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return otsu_img[1]

if __name__ == '__main__':
    img = cv2.imread(f'out/report/alpha_org.png', 0)
    img_rgb = cv2.imread(f'out/report/alpha_org.png')

    img_std_thresh = simple_thresholding(img)
    img_adapt_thresh = adaptive_thresholding(img)
    img_otsu = otsu_report(img)
    img_grabcut = grab_cut(img_rgb)
    img_watershed = watershed(img_rgb)

    cv2.imwrite("out/report/alpha-simple-thresholding.png", img_std_thresh)
    cv2.imwrite("out/report/alpha-adaptive-thresholding.png", img_adapt_thresh)
    cv2.imwrite("out/report/alpha-otsu-binarization.png", img_otsu)
    cv2.imwrite("out/report/alpha-grabcut.png", img_grabcut)
    cv2.imwrite("out/report/alpha-watershed.png", img_watershed)

    img = cv2.imread(f'out/report/epsilon_org.png', 0)
    img_rgb = cv2.imread(f'out/report/epsilon_org.png')

    img_std_thresh = simple_thresholding(img)
    img_adapt_thresh = adaptive_thresholding(img)
    img_otsu = otsu_report(img)
    img_grabcut = grab_cut(img_rgb)
    img_watershed = watershed(img_rgb)

    cv2.imwrite("out/report/epsilon-simple-thresholding.png", img_std_thresh)
    cv2.imwrite("out/report/epsilon-adaptive-thresholding.png", img_adapt_thresh)
    cv2.imwrite("out/report/epsilon-otsu-binarization.png", img_otsu)
    cv2.imwrite("out/report/epsilon-grabcut.png", img_grabcut)
    cv2.imwrite("out/report/epsilon-watershed.png", img_watershed)

