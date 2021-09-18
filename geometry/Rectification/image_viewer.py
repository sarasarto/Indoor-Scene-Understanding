import matplotlib.pyplot as plt
import math
import cv2
import numpy as np


class ImageViewer:
    def __init__(self, img_num=None, cols=4, axis=False):
        self.img_num = img_num
        self.cols = cols
        self.axis = axis
        self.counter = 0
        self.images = []
        self.builded = False

    def add(self, img, title='', cmap='rgb'):
        img_dict = {
            'img': img,
            'title': title,
            'cmap': cmap
        }
        self.images.append(img_dict)

    def remove_axis_values(self):
        self.axis = False

    def __len__(self):
        return self.img_num if not self.img_num is None else len(self.images)

    def __add_image(self, img, title='', cmap='rgb'):
        if cmap.lower() == 'bgr':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        y = self.counter % self.cols
        x = self.counter // self.cols
        self.counter += 1
        try:
            if self.img_num <= self.cols:
                self.axarr[y].imshow(img)
                self.axarr[y].set_title(title)
            else:
                self.axarr[x, y].imshow(img)
                self.axarr[x, y].set_title(title)
        except IndexError:
            print('Failed to access the plot [{}, {}]'.format(x, y))

    def __add_image_single(self, img, title='', cmap='rgb'):
        if cmap.lower() == 'bgr':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.axarr.imshow(img)
        self.axarr.set_title(title)

    def __remove_axis(self):
        for plot in self.axarr.flatten():
            plot.set_yticklabels([])
            plot.set_xticklabels([])

    def __remove_axis_single(self):
        self.axarr.set_yticklabels([])
        self.axarr.set_xticklabels([])

    def build(self):
        if not self.builded:
            self.img_num = len(self.images) if self.img_num is None else self.img_num
            self.cols = self.img_num if self.img_num < self.cols else self.cols
            self.fig, self.axarr = plt.subplots(math.ceil(self.img_num / self.cols), self.cols)
            if isinstance(self.axarr, np.ndarray):
                if not self.axis:
                    self.__remove_axis()
                for img_dict in self.images:
                    self.__add_image(**img_dict)
            else:
                if not self.axis:
                    self.__remove_axis_single()
                for img_dict in self.images:
                    self.__add_image_single(**img_dict)
            self.builded = True

    def show(self):
        self.build()
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state('zoomed')
        except AttributeError:
            mng.window.showMaximized()
        # mng.full_screen_toggle()
        plt.show()


def show_me(img, cmap='bgr', title=''):
    iv = ImageViewer()
    iv.add(img, title=title, cmap=cmap)
    iv.show()

