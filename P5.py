import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from sklearn.externals import joblib
from scipy.ndimage.measurements import label

from p5lib import window
from p5lib.display_tools import DisplayWindow
from p5lib.constants import *

# Global flags
debug = True
short = False
video = True

pyramid = [((64, 64),  [400, 500]),
           ((96, 96),  [400, 500]),
           ((128, 128),[400, 700])]

trace_size = 5
heatmap_trace =[]
boxes_trace = []

pipeline = joblib.load('model.pkl')

display_window = DisplayWindow(debug=debug)

prev_heatmap = None

def vehicle_detection(image):
    global display_window
    global prev_labels

    display_window.set_region('p1', image)

    draw_image = np.copy(image)

    windows = []
    for p in pyramid:
        windows += window.slide(image, x_start_stop=[None, None], y_start_stop=p[1], 
                            xy_window=p[0], xy_overlap=(0.5, 0.5))
                        
    hot_windows, prob_windows = window.search(image, windows, pipeline, color_space=COLOR_SPACE, 
                            spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, 
                            orient=ORIENT, pix_per_cell=PIX_PER_CELL, 
                            cell_per_block=CELL_PER_BLOCK, 
                            hog_channel=HOG_CHANNEL,
                            spatial_feat=SPATIAL_FEAT, hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)

    window_img = window.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    heatmap = window.generate_heatmap(draw_image, hot_windows, prob_windows, (1-1e-9))

    display_window.set_region('p2', heatmap)

    if len(heatmap_trace) < trace_size:
        heatmap_trace.append(heatmap)
    else:
        heatmap_trace[0:4], heatmap_trace[4] = heatmap_trace[1:5], heatmap
        heatmap = np.dstack(heatmap_trace)
        heatmap = np.average(heatmap, axis=2)

        display_window.set_region('p3', heatmap)

        heatmap[heatmap < (1-1e-10)] = 0
        prev_heatmap = heatmap

        display_window.set_region('p4', heatmap)

        labels = label(heatmap)

        # If no labels where found, use the previous
        if not labels[1] and prev_heatmap is not None:
            labels = label(prev_heatmap)
        elif labels[1]:
            prev_heatmap = heatmap

        final = window.draw_labeled_bboxes(np.copy(image), labels)

        display_window.set_region('p1', final)

    return display_window.get_output()



if not video:
    for img in os.listdir('test_images'):
        image = imread('test_images/' + img)
        plt.imshow(vehicle_detection(image))
        plt.show()
else:
    # video processing
    from moviepy.editor import VideoFileClip

    if short:
        video_input = 'test_video.mp4'
    else:
        video_input = 'project_video.mp4'

    video_output = video_input.replace('_video', '_output')

    clip = VideoFileClip(video_input)
    project_clip = clip.fl_image(vehicle_detection) #NOTE: this function expects color images!!
    project_clip.write_videofile(video_output, audio=False)
