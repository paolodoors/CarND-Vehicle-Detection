import os
import pickle
import glob
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from skimage.io import imread
from p5lib.constants import *
from p5lib import feature_extraction

data_file = 'raw-data-{}.p'.format(COLOR_SPACE)

if os.path.isfile(data_file):
    data = pickle.load(open(data_file, 'rb'))
    cars = data['cars']
    notcars = data['notcars']
else:
    # Divide up into cars and notcars
    pattern = os.path.join(DATA_DIR, '**' + os.sep + '*.png')
    images = glob.iglob(pattern, recursive=True)
    cars = []
    notcars = []
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(imread(image))
        elif 'vehicles' in image:
            cars.append(imread(image))

    # Pickle the images to avoid reading them each time
    pickle.dump({'cars': cars, 'notcars': notcars}, open(data_file, 'wb'))

# Extract the features for each image group: cars and notcars
car_features = feature_extraction.all_features(cars, color_space=COLOR_SPACE,
                                spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, hist_range=HIST_RANGE,
                                orient=ORIENT, pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, hog_channel=HOG_CHANNEL,
                                spatial_feat=SPATIAL_FEAT, hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)
notcar_features = feature_extraction.all_features(notcars, color_space=COLOR_SPACE,
                                spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, hist_range=HIST_RANGE,
                                orient=ORIENT, pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, hog_channel=HOG_CHANNEL,
                                spatial_feat=SPATIAL_FEAT, hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

pipeline = Pipeline([
    ('scaling', StandardScaler(with_mean=0, with_std=1)),
    ('classification', SVC(kernel='linear', probability=True))
    ])

pipeline.fit(X, y)
joblib.dump(pipeline, 'pipeline-{}.pkl'.format(COLOR_SPACE))
