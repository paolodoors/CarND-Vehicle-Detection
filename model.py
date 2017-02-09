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
    X = data['X']
    y = data['y']
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

    car_features = feature_extraction.all_features(cars, color_space=COLOR_SPACE, spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, hist_range=HIST_RANGE, spatial_feat=SPATIAL_FEAT, hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)
    notcar_features = feature_extraction.all_features(notcars, color_space=COLOR_SPACE, spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, hist_range=HIST_RANGE, spatial_feat=SPATIAL_FEAT, hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    pickle.dump({'X': X, 'y': y}, open(data_file, 'wb'))

pipeline = Pipeline([
    ('scaling', StandardScaler(with_mean=0, with_std=1)),
    ('classification', SVC(kernel='linear', probability=True))
    ])

pipeline.fit(X, y)
joblib.dump(pipeline, 'pipeline-{}.pkl'.format(COLOR_SPACE))

'''
seed = 123
kfold = KFold(n_splits=2, random_state=seed, shuffle=True)

t1 = time.time()

scores = cross_val_score(pipeline, X, y, cv=kfold)
avg_score = np.mean(scores)
print("Score: {}".format(avg_score))
print('{} seconds to train SVC with {} fold cross validation...'.format(time.time() - t1, kfold.get_n_splits()))

joblib.dump(pipeline, 'pipeline-{}-{}-fold-{:.2f}.pkl'.format(COLOR_SPACE, kfold.get_n_splits(), avg_score))
'''
