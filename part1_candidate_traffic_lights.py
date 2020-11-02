try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter
    from skimage.feature import peak_local_max

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt

    from skimage import color

except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")
"""
    k = grey_kernel.copy()
    for r in range(k.shape[0]):
        for c in range(k.shape[1]):
            if r > k.shape[0] / 6*5:
                k[r][c] = -100"""


#    grey_image = color.rgb2gray(c_image)
# red_kernel = kernel[:, :, 0]
# plt.imshow(im_convulation_res)
##plt.imshow(im_convulation_res3)

# x = np.arange(-100, 100, 20) + c_image.shape[1] / 2
# y_red = [c_image.shape[0] / 2 - 120] * len(x)
# y_green = [c_image.shape[0] / 2 - 100] * len(x)
# return x, y_red, x, y_green
def convolution(c_image: np.ndarray):
    kernel_drop_sky = np.array([-0.5, -0.5, -0.5,
                                -0.5, -0.5, -0.5,
                                -0.5, -0.5, -0.5,
                                1, 1, -0.5,
                                1, 2, 1,
                                1, 1, 1]).reshape(6, 3)
    return sg.convolve2d(c_image, kernel_drop_sky)


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    after_conv_red = convolution(c_image[:, :, 0])
    ####kernel = np.array(Image.open("kernel1.png"))
    # red_kernel=kernel[:, :, 1]
    ####grey_kernel = color.rgb2gray(color.rgba2rgb(kernel))
    ####im_convulation_res = sg.convolve2d(red_image, grey_kernel)
    ####print(kernel.shape)
    #### kernel_drop_sky = np.array([-1 / 9, -1 / 9, -1 / 9, -1 / 9, 8 / 9, 8 / 9, 8 / 9, 8 / 9, 8 / 9]).reshape(3, 3)

    ###im_drop_sky = sg.convolve2d(im_convulation_res,kernel_drop_sky)

    # plt.imshow(red_drop_sky)
    coordinates_red = peak_local_max(after_conv_red, min_distance=30, num_peaks=20)
    # print(type(coordinates_red))
    """
    kernel3 = np.array(Image.open("kernel3.png"))
    grey_kernel3 = color.rgb2gray(color.rgba2rgb(kernel))
    im_convulation_res3 = sg.convolve2d(im_convulation_res, grey_kernel3)"""

    ###im_filter_res = ndimage.maximum_filter(im_convulation_res, size=20)
    # im_filter_res = ndimage.maximum_filter(im_drop_sky, size=20)
    # plt.imshow(im_filter_res)

    # kernel4 = np.array(Image.open("g.png"))
    # grey_kernel4 = color.rgb2gray(color.rgba2rgb(kernel4))
    after_conv_green = convolution(c_image[:, :, 1])
    coordinates_green = peak_local_max(after_conv_green, min_distance=30, num_peaks=10)
    # im_filter_res4 = ndimage.maximum_filter(im_convulation_res4, size=20)
    # plt.imshow(im_filter_res4)

    # m = im_filter_res.max()
    # m2 = im_filter_res4.max()
    x_red = coordinates_red[:, 1] - 5
    y_red = coordinates_red[:, 0] - 5
    x_green = coordinates_green[:, 1]
    y_green = coordinates_green[:, 0] - 15
    """
    for r in range(im_filter_res.shape[0]):
        for c in range(im_filter_res.shape[1]):
            if im_filter_res[r][c] in list(np.arange(m - 5, m + 5)):
                x_red.append(c)
                y_red.append(r)
            if im_filter_res4[r][c] in list(np.arange(m2 - 5, m2 + 5)):
                x_green.append(c)
                y_green.append(r)"""

    return x_red, y_red, x_green, y_green


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)


"""    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()"""


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects,fig_num)
    #show_image_and_gt(image, None)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = "scaleup_imgs"
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        #test_find_tfl_lights(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
