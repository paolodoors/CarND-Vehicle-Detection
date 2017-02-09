# Directories
EXAMPLES_DIR = 'img'
DATA_DIR = 'data'

# Spatial features
SPATIAL_SIZE = (16, 16)

# Histogram features
HIST_BINS = 16
HIST_RANGE = (0, 256)

# Color histogram
COLOR_SPACE = 'HLS'

# HOG
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = 0

# What features to compute
SPATIAL_FEAT = False
HIST_FEAT = False
HOG_FEAT = True

# Discrimination
PROB_THRESHOLD = 0.6
