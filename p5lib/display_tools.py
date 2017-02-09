import cv2
import numpy as np
import matplotlib.pyplot as plt

class DisplayWindow():
    '''
           0                      960                    1920
    0      +------------------------+-----------------------+ 0
           |                        |                       |
           |                        |                       |
           |                        |                       |
           |          P1            |        P2             |
           |                        |                       |
           |                        |                       |
    540    +------------------------+-----------------------+ 540
           |                        |                       |
           |                        |                       |
           |                        |                       |
           |          P3            |        P4             |
           |                        |                       |
           |                        |                       |
    1080   +------------------------+------+----------------+ 1080
           0                     960                    1920
    '''

    sections = {'p1': [(0, 0),     (960, 540)],
                'p2': [(0, 960),   (960, 540)],
                'p3': [(540, 0),   (960, 540)],
                'p4': [(540, 960), (960, 540)]}

    def __init__(self, debug=False):
        self.screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.debug = debug
        self.cmap = plt.get_cmap('jet')

    def __set_image(self, image, size):
        if len(image.shape) < 3:
            image = np.stack((image,) * 3, axis=2) * 255
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    def set_region(self, region, params):
        if region not in self.sections:
            print ("Error, section {} doesn't exists".format(region))
            return None

        (y, x) = self.sections[region][0]
        (width, height) = self.sections[region][1]

        if region in ['p1', 'p2', 'p3', 'p4']:
            if region in ['p1'] or self.debug:
                self.screen[y:y+height, x:x+width] = self.__set_image(params, (width, height))
                

    def get_output(self):
        if self.debug:
            display = self.screen
        else:
            (y, x) = self.sections['p1'][0]
            (width, height) = self.sections['p1'][1]
            display = self.screen[y:y+height, x:x+width]

        return display
