import model
import streamlit as st
import utils
from visualizing_image import SingleImageViz

URL = "https://vqa.cloudcv.org/media/test2014/COCO_test2014_000000262567.jpg"
OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"

# load object, attribute, and answer labels
objids = utils.get_data(OBJ_URL)
attrids = utils.get_data(ATTR_URL)
vqa_answers = utils.get_data(VQA_URL)


@st.cache(allow_output_mutation=True)
def load_model():
    return model.Model()


@st.cache(allow_output_mutation=True)
def process_image(url):
    # image viz
    frcnn_visualizer = SingleImageViz(url, id2obj=objids, id2attr=attrids)
    # run frcnn
    images, sizes, scales_yx = qa_model.image_preprocess(url)
    output_dict = qa_model.cnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=qa_model.config.max_detections,
        return_tensors="pt",
    )
    # add boxes and labels to the image

    frcnn_visualizer.draw_boxes(
        output_dict.get("boxes"),
        output_dict.pop("obj_ids"),
        output_dict.pop("obj_probs"),
        output_dict.pop("attr_ids"),
        output_dict.pop("attr_probs"),
    )

    # Very important that the boxes are normalized
    normalized_boxes = output_dict.get("normalized_boxes")
    features = output_dict.get("roi_features")
    return {"normalized_boxes": normalized_boxes, "features": features}


st.title("Visual Question Answering")
st.markdown(
    "Based on [LXMERT: Learning Cross-Modality Encoder Representations from Transformers (Hao Tan, Mohit Bansal)](https://arxiv.org/abs/1908.07490)"
)

image_url = st.sidebar.text_input("Image url", URL)
if image_url:
    st.image(image_url, width=640)

    qa_model = load_model()

    image_features = process_image(image_url)

    # run lxmert
    question = st.text_input("Ask a question")

    if question and question != "":
        inputs = qa_model.tokenizer(
            question,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        output_vqa = qa_model.vqa(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=image_features["features"],
            visual_pos=image_features["normalized_boxes"],
            token_type_ids=inputs.token_type_ids,
            return_dict=True,
            output_attentions=False,
        )
        # get prediction
        pred_vqa = output_vqa["question_answering_score"].argmax(-1)
        st.text(vqa_answers[pred_vqa])
