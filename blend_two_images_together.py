import os
import cv2
import numpy as np

LEVEL = 8  # the number of levels in the pyramid


def laplacian_pyr(image, output_folder: str, get_lap_pyramid: bool = True) -> list:
    """
    :param image: an cv2 image
    :param output_folder: a path for your outputs folder
    :param get_lap_pyramid: a bool flag. If true will save the pyramid images in the output_folder
    :return: list of the laplacian pyramid of the given image.
    """
    if get_lap_pyramid:
        os.makedirs(output_folder, exist_ok=True)
    layer = np.copy(image)
    gaussian_pyr = [layer]
    for i in range(LEVEL):
        layer = cv2.pyrDown(layer)
        gaussian_pyr.append(layer)

    laplacian_pyramid = [gaussian_pyr[-1]]
    for i in range(LEVEL-1, 0, -1):
        layer_up = cv2.pyrUp(gaussian_pyr[i])
        layer_up = cv2.resize(layer_up, gaussian_pyr[i-1].shape[:2])
        laplacian = cv2.subtract(gaussian_pyr[i-1], layer_up)
        if get_lap_pyramid:
            cv2.imwrite(os.path.join(output_folder, f"{i}.jpg"), 255 * laplacian)
        laplacian_pyramid.append(laplacian)

    return laplacian_pyramid


def merge_half_of_laplacian_pyr(laplacian_pyr1: list, laplacian_pyr2: list):
    """
    :param laplacian_pyr1: first laplacian_pyramid list
    :param laplacian_pyr2: second laplacian_pyramid list
    :return: a list of the merging images of the laplacian pyramid of each image
    """
    merged_list = list()
    n = 0
    for lap1, lap2 in zip(laplacian_pyr1, laplacian_pyr2):
        n = n+1
        rows, cols, ch = lap1.shape
        merged_laplacian = np.hstack((lap1[:, : cols//2], lap2[:, cols//2:]))
        merged_list.append(merged_laplacian)

    return merged_list


def reconstruct_images(merged_lap_list: list):
    """
    :param merged_lap_list: list of images
    :return: image that was created from adding all the image in the list to each other
    """
    images_reconstruct = merged_lap_list[0]
    for i in range(1, LEVEL):
        images_reconstruct = cv2.pyrUp(images_reconstruct)
        images_reconstruct = cv2.resize(images_reconstruct, merged_lap_list[i].shape[:2])
        images_reconstruct = cv2.add(merged_lap_list[i], images_reconstruct)

    return images_reconstruct


def merge_two_images(image1_path: str, image2_path: str, output_folder: str, get_lap_pyramid: bool = True):
    """
    takes 2 images and smoothly blend each half of them together and saves the results
    :param image1_path: path to the first image
    :param image2_path:path to the second image
    :param output_folder: path to a folder where all the results will be saved in
    :param get_lap_pyramid: a bool flag. If true will save the pyramid images in the output_folder
    """
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    cv2.imwrite(os.path.join(output_folder, os.path.basename(image1_path)), image1)
    cv2.imwrite(os.path.join(output_folder, os.path.basename(image2_path)), image2)
    simple_merge = np.hstack((image1[:, : image1.shape[1] // 2], image2[:, image2.shape[1] // 2:]))
    cv2.imwrite(os.path.join(output_folder, 'simple_marge.jpg'), simple_merge)
    image1_laplacian_pyr_folder = os.path.join(output_folder, 'image1_laplacian_pyr')
    image1_laplacian_pyr = laplacian_pyr(image1/255, image1_laplacian_pyr_folder, get_lap_pyramid)
    image2_laplacian_pyr_folder = os.path.join(output_folder, 'image2_laplacian_pyr')
    image2_laplacian_pyr = laplacian_pyr(image2/255, image2_laplacian_pyr_folder, get_lap_pyramid)
    images_merged_laplacian = merge_half_of_laplacian_pyr(image1_laplacian_pyr, image2_laplacian_pyr)
    smooth_merged = reconstruct_images(images_merged_laplacian)
    cv2.imwrite(os.path.join(output_folder, 'smooth_marge.jpg'), 255 * smooth_merged)




