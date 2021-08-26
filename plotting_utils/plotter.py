import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

class Plotter():
    def random_colors(self, num_objs):
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(num_objs)]
        return colors


    def show_bboxes_and_masks(self, image, boxes, masks, labels, scores, output_file=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        alpha=0.5
        colors = self.random_colors(len(masks))

        for i,mask,box,label,score in zip(range(len(masks)), masks, boxes, labels, scores):
            
            #define bbox
            #(x1, y1) is top left
            #(x2,y2) is bottom right
            
    
            tl = 2  # line thickness
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            #color = random.choice(colors)

            cv2.rectangle(image, c1, c2, colors[i], tl)
            # draw text
            display_txt = "%s: %.1f%%" % (label, 100 * score)
            tf = 1  # font thickness
            t_size = cv2.getTextSize(display_txt, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, colors[i], -1)  # filled
            cv2.putText(image, display_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                        lineType=cv2.LINE_AA)

            #define mask
            for c in range(3):
                image[:, :, c] = np.where(mask == 1,
                                        image[:, :, c] *
                                        (1 - alpha) + alpha * colors[i][c],
                                        image[:, :, c])
        plt.imshow(image)
        plt.show()
        if output_file:
            plt.imsave(output_file, image)