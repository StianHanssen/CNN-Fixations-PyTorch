"""
===================
Image Slices Viewer
===================

Scroll through 2D image slices of a 3D array.
Credit: https://matplotlib.org/gallery/event_handling/image_slices_viewer.html
Modified version
"""
import numpy as np
import matplotlib.pyplot as plt

from cnn_fixations import utils as U


class IndexTracker(object):
    def __init__(self, ax, image, points, heatmap):
        self.ax = ax
        self.vmin, self.vmax = image.min(), image.max()

        self.image = image
        self.points = points
        self.heatmap = heatmap
        self.slices = image.shape[0]
        self.ind = self.slices//2

        self.im = ax[0].imshow(self.image[self.ind, :, :], cmap='gray', vmin=self.vmin, vmax=self.vmax)
        ax[0].set_title('Image')

        self.background_points = ax[1].imshow(self.image[self.ind, :, :], cmap='gray', vmin=self.vmin, vmax=self.vmax)
        self.scatter_points = (ax[1].scatter(self.points[self.ind][:, 0], points[self.ind][:, 1], c='r')
                               if len(self.points[self.ind])
                               else ax[1].scatter([], [], c='r'))
        ax[1].axis('off')
        ax[1].set_title('CNN Fixations')

        self.background_heatmap = ax[2].imshow(self.image[self.ind, :, :], cmap='gray', vmin=self.vmin, vmax=self.vmax)
        self.im_heatmap = ax[2].imshow(self.heatmap[self.ind, :, :], 'jet', alpha=0.6)
        ax[2].axis('off')
        ax[2].set_title('Localization Map')

        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.image[self.ind, :, :])

        self.background_points.set_data(self.image[self.ind, :, :])
        if len(self.points[self.ind]):
            self.scatter_points.set_offsets(np.c_[self.points[self.ind][:, 1], self.points[self.ind][:, 0]])
        else:
            self.scatter_points.set_offsets(np.c_[[], []])

        self.background_heatmap.set_data(self.image[self.ind, :, :])
        self.im_heatmap.set_data(self.heatmap[self.ind, :, :])

        self.ax[0].set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


# Input shape (depth, height, width)
def display_3d_image(image, points, heatmap, image_label, prediction, case_num=None):
    fig, ax = plt.subplots(1, 3, figsize=(13, 5))
    tracker = IndexTracker(ax, image, points, heatmap)
    if image_label is not None:
        image_label = 'AMD' if image_label.item() else 'Control'
    else:
        image_label = 'Unknown'
    prediction = 'AMD' if prediction.item() else 'Control'
    if case_num is not None:
        fig.suptitle(f'Prediction: {prediction} | True Label: {image_label} | Case Index: {case_num}')
    else:
        fig.suptitle(f'Prediction: {prediction} | True Label: {image_label}')
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    U.create_colorbar(fig, heatmap, tracker.im_heatmap, tracker.ax)
    plt.show()
