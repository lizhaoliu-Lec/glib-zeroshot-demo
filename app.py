import requests
import os
from io import BytesIO
from PIL import Image
import numpy as np
from pathlib import Path
import gradio as gr

import warnings

warnings.filterwarnings("ignore")

# os.system(
#     "pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo click opencv-python inflect nltk scipy scikit-learn pycocotools")
# os.system("pip install transformers")
os.system("python setup.py build develop --user")

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

# Use this command for evaluate the GLIP-T model
config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"

# Use this command if you want to try the GLIP-L model
# ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O MODEL/glip_large_model.pth
# config_file = "configs/pretrain/glip_Swin_L.yaml"
# weight_file = "MODEL/glip_large_model.pth"

# update the config options with the config file
# manual override some options
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)


def predict(image, text):
    result, _ = glip_demo.run_on_web_image(image[:, :, [2, 1, 0]], text, 0.5)
    return result[:, :, [2, 1, 0]]


image = gr.inputs.Image()

gr.Interface(
    description="Object Detection in the Wild through GLIP (https://github.com/microsoft/GLIP).",
    fn=predict,
    inputs=["image", "text"],
    outputs=[
        gr.outputs.Image(
            type="pil",
            # label="grounding results"
        ),
    ],
    examples=[
        ["./flickr_9472793441.jpg", "bobble heads on top of the shelf ."],
        ["./flickr_9472793441.jpg", "sofa . remote . dog . person . car . sky . plane ."],
        ["./coco_000000281759.jpg", "A green umbrella. A pink striped umbrella. A plain white umbrella."],
        ["./coco_000000281759.jpg", "a flowery top. A blue dress. An orange shirt ."],
        ["./coco_000000281759.jpg", "a car . An electricity box ."],
        ["./flickr_7520721.jpg", "A woman figure skater in a blue costume holds her leg by the blade of her skate ."]
    ],
    article=Path("docs/intro.md").read_text()
).launch()
# ).launch(server_name="0.0.0.0", server_port=7000, share=True)
