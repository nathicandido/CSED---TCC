import cv2
import numpy

def lk(image):
    myimg = cv2.imread(image)
    avg_color_per_row = numpy.average(myimg, axis=0)
    avg_color = numpy.average(avg_color_per_row, axis=0)
    print(avg_color)
