from retrieval.evaluation import RetrievalMeasure


class Evaluator():
    def __init__(self, query_img, results) -> None:
        self.query_img = query_img
        self.results = results
        self.rm = RetrievalMeasure()

    def eval(self):
        question = input('Do you want to evaluate the method?? [y/n]: ')
        if question == 'y':
            # starting evaluation
            user_resp = self.rm.get_user_relevance_retrieval(self.results, self.query_img)
            print("user responses:")
            print(user_resp)
            single_AP = self.rm.get_AP(user_resp, 5)
            print("Average Precision: " + str(single_AP))
