import cv2


def pad(image, expected_size=200):
    height, width = image.shape

    white = (255, 255, 255)

    delta_h = expected_size - height
    delta_w = expected_size - width
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=white)

    return padded_image
