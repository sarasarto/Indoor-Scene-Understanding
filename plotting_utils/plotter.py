import random
import cv2
import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec
import numpy as np


class Plotter():
    def plot_image(self, image, title=None):
        plt.imshow(image)
        plt.title(title)
        plt.show()

    def plot_imgs_by_row(self, images: list, titles: list, num_imgs):
        f, axes = plt.subplots(1, num_imgs)

        for i, image, title in zip(range(num_imgs), images, titles):
            axes[i].set_title(title)
            axes[i].imshow(image)
        plt.show()

    def random_colors(self, num_objs):
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(num_objs)]
        return colors

    def show_bboxes_and_masks(self, image, boxes, masks, labels, scores, output_file=None):
        image = np.copy(image)
        f, axarr = plt.subplots(1, 2)
        alpha = 0.5
        colors = self.random_colors(len(masks))

        axarr[0].set_title('Test image')
        axarr[0].imshow(image)

        for i, mask, box, label, score in zip(range(len(masks)), masks, boxes, labels, scores):

            # define bbox
            # (x1, y1) is top left
            # (x2,y2) is bottom right

            tl = 2  # line thickness
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

            cv2.rectangle(image, c1, c2, colors[i], tl)
            # draw text
            display_txt = "%s: %.1f%%" % (label, 100 * score)
            tf = 1  # font thickness
            t_size = cv2.getTextSize(display_txt, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, colors[i], -1)  # filled
            cv2.putText(image, display_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                        lineType=cv2.LINE_AA)

            # define mask
            for c in range(3):
                image[:, :, c] = np.where(mask == 1,
                                          image[:, :, c] *
                                          (1 - alpha) + alpha * colors[i][c],
                                          image[:, :, c])

        axarr[1].set_title('Segmented image')
        axarr[1].imshow(image)
        plt.show()
        if output_file:
            plt.imsave(output_file, image)

    def plot_retrieval_results(self, query_img, similar_images: list, retrieval_method):
        for img in similar_images:
            img = img.astype('float32')

        # images are already sorted by similarity
        # we always plot first 5 results
        fig = plt.figure()
        fig.suptitle(f'Retrieval results with {retrieval_method} method')

        gs = GridSpec(2, 5)
        query_axis = fig.add_subplot(gs[0, 1])
        query_axis.set_title('Query image')
        query_axis.imshow(cv2.resize(query_img, (224, 224)))

        ax1 = fig.add_subplot(gs[1, 0])
        ax1.imshow(cv2.resize(similar_images[0], (224, 224)))

        ax2 = fig.add_subplot(gs[1, 1])
        ax2.imshow(cv2.resize(similar_images[1], (224, 224)))

        ax3 = fig.add_subplot(gs[1, 2])
        ax3.imshow(cv2.resize(similar_images[2], (224, 224)))
        ax3.set_title('Results:')

        ax4 = fig.add_subplot(gs[1, 3])
        ax4.imshow(cv2.resize(similar_images[3], (224, 224)))

        ax5 = fig.add_subplot(gs[1, 4])
        ax5.imshow(cv2.resize(similar_images[4], (224, 224)))

        plt.show()

    def plot_evaluation(self, query_img, result):
        fig = plt.figure(1, figsize=(10, 10))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(query_img)
        ax1.set_title('Query')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(result)
        ax2.set_title('Retrieved Image')
        plt.suptitle("Evaluation")

        plt.show()

    def plot_loss_values(self, loss_tensor):
        #loss tensor has shape (10, 2119)--> 2119 == num of iterations in each epoch, 
        #considering batch_size == 2
        loss_tensor = loss_tensor[:10,:] #consider first 10 epochs
        mean_losses = torch.mean(loss_tensor, 1)
        plt.plot(mean_losses)
        plt.xticks(range(10))
        plt.show()

        import matplotlib.pyplot as plt
import torch
import numpy as np

np.random.seed(0)
a = np.random.randint(low=0, high=50, size=100)

plt.xticks(np.arange(0, len(a)+1, 20))
plt.plot(a)
plt.show()