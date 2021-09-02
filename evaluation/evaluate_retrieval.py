import argparse
from plotting_utils.plotter import Plotter
from retrieval.retrieval_manager import ImageRetriever
from retrieval.method_dhash.helper_DHash import DHashHelper
from retrieval.method_SIFT.helper_SIFT import SIFTHelper
from retrieval.method_autoencoder.helper_autoenc import AutoencHelper
from eval_manager import Evaluator
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Retrieval evaluation')

    parser.add_argument('-retr', '--retrieval', type=str,
                        help='Type of retrieval method (sift/ dhash / autoencoder)', required=True)

    args = parser.parse_args()
    retr_method = args.retrieval

    if retr_method not in ['sift', 'dhash', 'autoencoder']:
        raise ValueError('Model type must be \'sift\' or \'dhash\' or \'autoencoder\'')

    if retr_method == 'sift':
        helper = SIFTHelper()
    else:
        if retr_method == 'dhash':
            helper = DHashHelper()
        else:
            helper = AutoencHelper()

    retrieval_classes = []
    with open('retrieval/retrieval_classes.txt') as f:
        retrieval_classes = f.read().splitlines()

    img_retriever = ImageRetriever(helper)

    

    if retr_method == 'sift':
        results = img_retriever.find_similar_furniture(res_img, label)
    if retr_method == 'dhash':
        PIL_image = Image.fromarray(np.uint8(query_img)).convert('RGB')
        results = img_retriever.find_similar_furniture(PIL_image, label)
    if retr_method == 'autoencoder':
        results = img_retriever.find_similar_furniture(Image.fromarray(query_img), label)

    pt = Plotter()
    pt.plot_retrieval_results(query_img, results, retr_method)
    img_evaluator = Evaluator(query_img, results)
    img_evaluator.eval()

if __name__ == '__main__':
    main()