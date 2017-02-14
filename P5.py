import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from skimage.io import imread
from sklearn.externals import joblib
from scipy.ndimage.measurements import label

from p5lib import window
from p5lib.display_tools import DisplayWindow
from p5lib.constants import *

# Global flags
debug = False

pyramid = [((48, 48),  [None, None], [400, 500], (0.5, 0.5)),
           ((64, 64),  [None, None], [400, 500], (0.5, 0.5)),
           ((96, 96),  [None, None], [400, 500], (0.5, 0.5)),
           ((128, 128),[None, None], [400, 600], (0.5, 0.5))]

trace_size = TRACE
heatmap_trace =[]
boxes_trace = []

display_window = DisplayWindow(debug=debug)

def vehicle_detection(image):
    global display_window
    global prev_labels

    display_window.set_region('p1', image)

    draw_image = np.copy(image)

    windows = []
    for p in pyramid:
        windows += window.slide(image, x_start_stop=p[1], y_start_stop=p[2], xy_window=p[0], xy_overlap=p[3])
                        
    hot_windows, prob_windows = window.search(image, windows, pipeline, color_space=config['COLOR_SPACE'], threshold=PROB_THRESHOLD_DETECTION,
                            spatial_size=config['SPATIAL_SIZE'], hist_bins=config['HIST_BINS'], hist_range=config['HIST_RANGE'],
                            orient=config['ORIENT'], pix_per_cell=config['PIX_PER_CELL'], cell_per_block=config['CELL_PER_BLOCK'], hog_channel=config['HOG_CHANNEL'],
                            spatial_feat=config['SPATIAL_FEAT'], hist_feat=config['HIST_FEAT'], hog_feat=config['HOG_FEAT'])

    heatmap = window.generate_heatmap(draw_image, hot_windows, prob_windows, PROB_THRESHOLD_DETECTION)

    display_window.set_region('p2', heatmap)

    if len(heatmap_trace) < trace_size:
        heatmap_trace.append(heatmap)
    else:
        # Create a FIFO fixed list to track the heatmaps
        heatmap_trace[0:trace_size-1], heatmap_trace[trace_size-1] = heatmap_trace[1:trace_size+1], heatmap
        heatmap = np.dstack(heatmap_trace)
        w = np.exp(np.arange(1/trace_size, 1.01, 1/trace_size) * 2)
        heatmap = np.average(heatmap, axis=2, weights=w)

        display_window.set_region('p3', heatmap)

        heatmap[heatmap < PROB_THRESHOLD_FILTER] = 0

        display_window.set_region('p4', heatmap)

        labels = label(heatmap)

        final = window.draw_labeled_bboxes(image, labels)

        display_window.set_region('p1', final)

    return display_window.get_output()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='model used to classify cars')
    parser.add_argument('video', help='which video to use, short or long')
    args = parser.parse_args()

    data = joblib.load(args.model)
    pipeline = data['model']
    config = data['config']

    # video processing
    from moviepy.editor import VideoFileClip

    if args.video == 'short':
            video_input = 'test_video.mp4'
    else:
        video_input = 'project_video.mp4'

    video_output = 'output_detection_'+args.video+'_'+str(PROB_THRESHOLD_DETECTION)+'_filter_'+str(PROB_THRESHOLD_FILTER)+'.mp4'

    clip = VideoFileClip(video_input)
    project_clip = clip.fl_image(vehicle_detection) #NOTE: this function expects color images!!
    project_clip.write_videofile(video_output, audio=False)
