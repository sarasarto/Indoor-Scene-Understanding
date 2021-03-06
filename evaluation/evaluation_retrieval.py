import numpy as np
from plotting_utils.plotter import Plotter


class RetrievalMeasure():
    def __init__(self) -> None:
        self.pt = Plotter()
        pass

    # this fuction shows the retrieved images to the user that says if they are relevant or not
    # return: the user response
    def get_user_relevance_retrieval(self, images, query):

        print("Starting eval user relevance... ")
        user_responses = []

        for i in range(len(images)):
            self.pt.plot_evaluation(query, images[i])
            result = input('Si tratta di un\' immagine rilevante? [y/n]: ')
            if result == 'y':
                user_responses.append(1)
            else:
                user_responses.append(0)
        user_responses = np.array(user_responses)

        return user_responses

    # return: the AP value
    def get_AP(self, user_responses, rank):
        precisions = []
        num_cur_relevant = 0
        for i in range(rank):
            if user_responses[i] == 1:
                num_cur_relevant += 1
                precisions.append(num_cur_relevant / (i + 1))

        return np.mean(np.array(precisions))

    def compute_MAP(self, AP_vector):
        return np.mean(AP_vector)
