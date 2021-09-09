from evaluate_retrieval import RetrievalMeasure


class Evaluator():
    def __init__(self, query_img, results) -> None:
        self.query_img = query_img
        self.results = results
        self.rm = RetrievalMeasure()

    def eval(self):

        # starting evaluation
        user_resp = self.rm.get_user_relevance_retrieval(self.results, self.query_img)
        print("user responses:")
        print(user_resp)
        single_AP = self.rm.get_AP(user_resp, 5)
        print("Average Precision: " + str(single_AP))
        return single_AP

    def compute_MAP(self, AP_test):
        self.rm.compute_MAP(AP_test)