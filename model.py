from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
from utils import Config


class Model:
    def __init__(self):
        self.config = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")

        self.cnn = GeneralizedRCNN.from_pretrained(
            "unc-nlp/frcnn-vg-finetuned", config=self.config
        )

        self.image_preprocess = Preprocess(self.config)

        self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.vqa = LxmertForQuestionAnswering.from_pretrained(
            "unc-nlp/lxmert-vqa-uncased"
        )


# download model parameters
Model()
