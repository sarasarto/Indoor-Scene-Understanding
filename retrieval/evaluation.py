from matplotlib import use
import matplotlib.pyplot as plt
import numpy as np
from plotting_utils.plotter import Plotter


class RetrievalMeasure():
    def __init__(self) -> None:
        self.pt = Plotter()
        pass

    # questo metodo l'avevamo fatto all'inizio
    # perche passavamo la label???
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

    def get_user_relevance_retrieval(self, images, query):

        print("starting eval user relevance... ")
        user_responses = []
        #self.pt.plot_image(query, title='Query Image')
        for i in range(len(images)):
            #self.pt.plot_image(images[i], title='Evaluation of Retrieved Image')
            self.pt.plot_evaluation(query, images[i])
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
