import argparse
import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from gradio_image_prompter import ImagePrompter

from util.misc import nested_tensor_from_tensor_list
import datasets.transforms as T
from draw_box_utils import draw_objs
from util import box_ops
import matplotlib.pyplot as plt
from util.slconfig import SLConfig
from torchvision.ops.boxes import nms


config_py_file = 'config/cfg_odvg.py'
weights_path = 'hdino_t.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image_and_target(image_pil, box_prompt, class_id):
    w, h = image_pil.size

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    boxes = torch.as_tensor(box_prompt, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)
    classes = torch.tensor(class_id, dtype=torch.int64)

    target = {
        "boxes": boxes,
        "labels": classes,
        "orig_size": torch.as_tensor([int(h), int(w)]),
        "size": torch.as_tensor([int(h), int(w)])
    }

    image, target = transform(image_pil, target)
    target = [{k: v.to(device) for k, v in target.items()}]

    return image_pil, image, target


def build_model_main(config_file):

    from models.registry import MODULE_BUILD_FUNCS
    assert config_file.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(config_file.modelname)

    model = build_func(config_file)
    weights_dict = torch.load(config_file.weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(config_file.device)
    model = model.eval()
    return model


def get_output(model, image, device, max_class_id, box_threshold, iou_threshold, targets=None):
    with torch.no_grad():
        img = image.to(device)
        outputs = model(img[None], targets=targets, cap_list=targets[0]["cap_list"])

        pos_map = torch.eye(max_class_id+1, 256).to(device)
        logits = outputs["pred_logits"].sigmoid()[0] @ pos_map.T
        boxes = outputs["pred_boxes"][0]

        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold

        cls_filt = logits_filt.max(dim=1)[1][filt_mask]
        logits_filt = logits_filt.max(dim=1)[0][filt_mask]
        boxes_filt = boxes_filt[filt_mask]


        if len(boxes_filt) > 0:
            item_indices = nms(boxes_filt, logits_filt, iou_threshold=iou_threshold)
            boxes_filt = boxes_filt[item_indices]
            logits_filt = logits_filt[item_indices]
            cls_filt = cls_filt[item_indices]

        boxes_filt = box_ops.box_cxcywh_to_xyxy(boxes_filt)
        orig_target_sizes = targets[0]["orig_size"].unsqueeze(0)
        scale_fct = torch.stack([orig_target_sizes[:, 1], orig_target_sizes[:, 0],
                                 orig_target_sizes[:, 1], orig_target_sizes[:, 0]], dim=1).cpu()
        boxes_filt = boxes_filt * scale_fct

        return boxes_filt, logits_filt, cls_filt


def run_inference(raw_image, texts, model_id, box_threshold, iou_threshold):
    config_file = SLConfig.fromfile(config_py_file)
    config_file.device = device

    config_file.weights_path = weights_path
    model = build_model_main(config_file)

    texts = texts.split(' . ')
    category_index = {str(i): text for i, text in enumerate(texts)}
    image_pil = raw_image.convert("RGB")
    w, h = image_pil.size

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)
    image_tensor = image_tensor.to(device)
    max_class_id = len(texts)-1
    classes = torch.tensor(max_class_id, dtype=torch.int64).to(device)
    orig_size = torch.as_tensor([int(h), int(w)]).to(device)
    size = torch.as_tensor([int(h), int(w)]).to(device)
    target = [{
        "cap_list": texts,
        "labels": classes,
        "orig_size": orig_size,
        "size": size,
    }
]
    boxes_filt, logits_filt, cls_filt = get_output(model, image_tensor, device,max_class_id, box_threshold, iou_threshold, targets=target)

    try:
        plot_img = draw_objs(image_pil,
                             boxes=boxes_filt.numpy(),
                             scores=logits_filt.numpy(),
                             cls_filt=cls_filt.numpy(),
                             category_index=category_index,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        return plot_img

    except Exception as e:
        print(f"Error: {e}")
        return image_pil


def app():
    with gr.Blocks() as demo:
        gr.Markdown("## HDINO Object Detection Demo")
        with gr.Row():
            with gr.Column():

                raw_image = gr.Image(type="pil", label="raw image", visible=True)
                texts = gr.Textbox(label="text prompt", value="person")

                detect_btn = gr.Button("Detect")

                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "hdino_t",
                    ],
                    value="hdino_t",
                )


                box_thresh = gr.Slider(0.0, 1.0, 0.15, label="Confidence threshold", step=0.01)
                iou_thresh = gr.Slider(0.0, 1.0, 0.7, label="IoU threshold")

            with gr.Column():
                output_image = gr.Image(type="pil", label="Detection results")


        detect_btn.click(
            fn=run_inference,
            inputs=[raw_image, texts, model_id, box_thresh, iou_thresh],
            outputs=output_image
        )

    return demo


if __name__ == '__main__':
    demo = app()
    demo.launch(share=True)