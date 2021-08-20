from matplotlib import use
import matplotlib.pyplot as plt
import numpy as np


class RetrievalMeasure():
    def __init__(self) -> None:
        pass

    def get_user_relevance(class_label, query_image, images):
        plt.imshow(query_image)
        plt.title('Query image')
        plt.show()

        user_responses = []

        for i in len(images):
            plt.imshow(images[i])
            plt.show()

            result = input('Si tratta di un\' immagine rilevante? [y/n]: ')
            if result == 'y':
                user_responses.append(1)
            else:
                user_responses.append(0)
        user_responses = np.array(user_responses)

    def get_user_relevance_retrieval(self, images):

        print("starting eval user relevance... ")

        user_responses = []

        for i in range(len(images)):
            plt.imshow(images[i])
            plt.title('Evaluation of Retrieved Image')
            plt.show()

            result = input('Si tratta di un\' immagine rilevante? [y/n]: ')
            if result == 'y':
                user_responses.append(1)
            else:
                user_responses.append(0)
        user_responses = np.array(user_responses)

        return user_responses


    def get_AP(self, user_responses, rank):
        num_relevant_docs = len(user_responses[user_responses == 1])

        precisions = []
        num_cur_relevant = 0
        for i in range(rank):
            if user_responses[i] == 1:
                num_cur_relevant += 1
                precisions.append(num_cur_relevant / (i + 1))

        return np.mean(np.array(precisions))

    def compute_MAP(self, AP_vector):
        return np.mean(AP_vector)
