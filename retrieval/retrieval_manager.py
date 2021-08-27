from retrieval.SIFT_helper import SIFTHelper

class ImageRetriever():
    def __init__(self) -> None:
        pass

    def find_similar_furniture(query_img, label, retrieval_method):
        if retrieval_method not in ['sift', 'dhash', 'autoencoder']:
            raise ValueError('Unsupported retrieval method: try with [sift, dhash, autoencoder]!')
        
        if retrieval_method == 'sift':
            #call sift helper or similar things
            sift_helper = SIFTHelper()


        elif retrieval_method == 'dhash':
            #call dhash helper
            pass
            
        elif retrieval_method == 'autoencoder':
            #call autoenc helper
            pass
