import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################


def image_print(img):
    """
    Helper function to print out images, for debugging. Pass them in as a list.
    Press any key to continue.
    """
    src=cv2.imread(img)
    img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    cv2.imshow("image", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_color_scatter_RGB(img):
    src=cv2.imread(img)
    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    r,g,b = cv2.split(img)
    fig=plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()
    
def plot_color_scatter_HSV(img):
    src=cv2.imread(img)
    img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()

def show_range(img):
    
    light_orange = (1, 190, 200)
    dark_orange = (18, 255, 255)
    src=cv2.imread(img)
    img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, light_orange, dark_orange)
    result = cv2.bitwise_and(img, img, mask=mask)
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()

        

def cd_color_segmentation(img, template):
    """
    Implement the cone detection using color segmentation algorithm
    Input:
            img: np.3darray; the input image with a cone to be detected. BGR.
            template_file_path; Not required, but can optionally be used to automate setting hue filter values.
    Return:
            bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
                            (x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
    """
    ########## YOUR CODE STARTS HERE ##########

    bounding_box = ((0, 0), (0, 0))

    ########### YOUR CODE ENDS HERE ###########

    # Return bounding box
    return bounding_box


image_print("./test_images_cone/test2.jpg")
# plot_color_scatter_HSV("./test_images_cone/test4.jpg")
show_range("./test_images_cone/test2.jpg")