class ImageRetriever():
    def __init__(self, retrieval_helper) -> None:
        self.retrieval_helper = retrieval_helper

    def find_similar_furniture(self, query_img, label):
        similar_images = self.retrieval_helper.retrieval(query_img, label)
        return similar_images