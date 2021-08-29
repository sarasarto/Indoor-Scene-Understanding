class ImageRetriever():
    def __init__(self, retrieval_helper) -> None:
        self.retrieval_helper = retrieval_helper

    def find_similar_furniture(self, query_img, label):
        obj_list, num_good = self.retrieval_helper.retrieval(query_img, label)
        retr_imgs = self.retrieval_helper.print_results(obj_list, num_good, label)