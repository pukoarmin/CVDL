import glob
import logging
import numpy as np
import cv2
from tqdm import tqdm as progress_bar
import matplotlib.pyplot as plt

# Setup logger
logging.basicConfig(format="%(asctime)s - [MAIN] -> %(message)s", datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

# Load images from a folder
images_folder = glob.glob("../Data/goodcameras/*.png")

logging.info("Found " + str(len(images_folder)) + " images")

for image_path_index in progress_bar(range(len(images_folder))):
    image_path = images_folder[image_path_index]
    image = cv2.imread(image_path)
    logging.info("Processing image: " + image_path)

    # creating empty image of same size
    height, width, _ = image.shape
    empty_image = np.zeros((height, width), np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    val = 135
    val = val / 100  # dividing by 100 to get in range 0-1.5
    # scale pixel values up or down for channel 1(Saturation)
    hsv[:, :, 1] = hsv[:, :, 1] * val
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255  # setting values > 255 to 255.
    # scale pixel values up or down for channel 2(Value)
    hsv[:, :, 2] = hsv[:, :, 2] * val
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255  # setting values > 255 to 255.
    hsv = np.array(hsv, dtype=np.uint8)
    sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.namedWindow('saturated', cv2.WINDOW_NORMAL)
    cv2.imshow("saturated", sat)

    # APPLIED K-MEANS CLUSTERING
    Z = sat.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 6
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(image.shape)

    cv2.namedWindow('res2', cv2.WINDOW_NORMAL)
    cv2.imshow("res2", res2)

    # CONVERTED TO A LUV IMAGE AND MADE EMPTY IMAGE, A MASK
    blur = cv2.GaussianBlur(res2, (5, 5), cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #cl = clahe.apply(gray)
    LUV = cv2.cvtColor(blur, cv2.COLOR_RGB2LUV)
    l = LUV[:, :, 0]
    v1 = l > 80
    v2 = l < 150
    value_final = v1 & v2
    empty_image[value_final] = 255

    # APPLIED BITWISE-AND ON GRAYSCALE IMAGE AND EMPTY IMAGE TO OBTAIN ROAD AND SOME-OTHER IMAGES TOO
    final = cv2.bitwise_and(gray, empty_image)
    contours, hierchary = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final = cv2.drawContours(final, contours, -1, 0, 3)

    cv2.namedWindow('final', cv2.WINDOW_NORMAL)
    cv2.imshow('final', final)

    # FURTHER MASKED THE FINAL IMAGE TO OBTAIN ONLY THE ROAD PARTICLES
    final_masked = np.zeros((height, width), np.uint8)
    v1 = final >= 80
    v2 = final <= 150
    final_masked[v1 & v2] = 255

    cv2.namedWindow('final_masked', cv2.WINDOW_NORMAL)
    cv2.imshow('final_masked', final_masked)

    # APPLIED EROSION,CONTOURS AND TOP-HAT TO REDUCE NOISE
    kernel = np.ones((3, 3), np.uint8)
    final_eroded = cv2.erode(final_masked, kernel, iterations=1)
    contours, hierchary = cv2.findContours(final_eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_masked = cv2.drawContours(final_eroded, contours, -1, 0, 3)

    final_waste = cv2.morphologyEx(final_masked, cv2.MORPH_TOPHAT, kernel, iterations=2)
    final_waste = cv2.bitwise_not(final_waste)
    final_masked = cv2.bitwise_and(final_waste, final_masked)

    # USED FLOOD-FILL TO FILL IN THE SMALL BLACK LANES
    final_flood = final_masked.copy()
    h, w = final_masked.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(final_flood, mask, (0, 0), 255)
    final_flood = cv2.bitwise_not(final_flood)
    final_filled = cv2.bitwise_or(final_masked, final_flood)
    #final_filled = cv2.bitwise_not(final_filled)

    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2.imshow('original', image)
    cv2.namedWindow("generated", cv2.WINDOW_NORMAL)
    cv2.imshow("generated", final_filled)

    cv2.imwrite("D:\\UBB\\An 3\\CVDL\\FinalProject\\Dataset Gatherer\\Data\\results\\" + str(image_path_index) + ".png",
                final_filled)
    cv2.waitKey(1)
