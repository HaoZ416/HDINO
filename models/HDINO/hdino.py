# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)


import copy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel, AutoTokenizer

from util.misc import (
    NestedTensor,
    inverse_sigmoid,
    nested_tensor_from_tensor_list,
)

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone

from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss

from .aux_query_components import prepare_for_aux_query, aux_query_post_process


class HDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
            self,
            backbone,
            transformer,
            num_queries,
            aux_loss=False,
            iter_update=False,
            query_dim=2,
            num_feature_levels=1,
            nheads=8,
            # two stage
            two_stage_type="no",  # ['no', 'standard']
            dec_pred_bbox_embed_share=True,
            two_stage_class_embed_share=True,
            two_stage_bbox_embed_share=True,
            num_patterns=0,
            aux_query_number=100,
            aux_box_noise_scale=0.4,
            max_labels=80,
            text_encoder_type="",
            max_text_len=256,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 256
        self.one2many_embeddings = nn.Embedding(self.num_queries * 2, hidden_dim)

        self.max_labels = max_labels

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.aux_query_number = aux_query_number
        self.aux_box_noise_scale = aux_box_noise_scale

        self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_type)
        self.clip = CLIPTextModel.from_pretrained(text_encoder_type)

        self.feat_map = nn.Linear(self.clip.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        self.fusion_feat_map = nn.Linear(self.clip.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.fusion_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.fusion_feat_map.weight.data)

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def forward(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        device = samples.tensors.device

        cap_lists = kw["cap_list"]
        tokenized = self.tokenizer(cap_lists, padding=True, return_tensors="pt").to(device)
        clip_output = self.clip(**tokenized)

        cls_embeddings = clip_output.pooler_output.unsqueeze(0)

        encoded_text = self.feat_map(cls_embeddings)
        fusion_encoded_text = self.fusion_feat_map(cls_embeddings)
        text_mask = torch.ones((1, len(cap_lists)), dtype=torch.bool).to(device)

        bs, len_td, _ = encoded_text.shape


        features, poss = self.backbone(samples)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        if self.training and (self.aux_query_number > 0 or targets is not None):
            input_query_label, input_query_bbox, attn_mask, aux_query_meta = \
                prepare_for_aux_query(aux_query_args=(targets, self.aux_query_number, self.aux_box_noise_scale),
                                training=self.training, num_queries=self.num_queries, hidden_dim=self.hidden_dim,
                                one2many_embeddings=self.one2many_embeddings, device=device)
        else:
            input_query_bbox = input_query_label = attn_mask = aux_query_meta = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, encoded_text, text_mask, fusion_encoded_text
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, encoded_text, text_mask)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )

        if self.aux_query_number > 0 and aux_query_meta is not None:
            outputs_class, outputs_coord_list = \
                aux_query_post_process(outputs_class, outputs_coord_list,
                                aux_query_meta, self.aux_loss, self._set_aux_loss)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1]}
        out['aux_query_meta'] = aux_query_meta

        # Used to calculate losses
        if len_td < self.max_text_len:
            max_text_len = self.max_text_len
        else:
            max_text_len = len_td
        out['text_mask'] = torch.zeros(bs, max_text_len, dtype=torch.bool).to(device)
        out['text_mask'][:, :len_td] = text_mask
        out["encoded_text"] = encoded_text

        # for intermediate outputs
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], encoded_text, text_mask)
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]



@MODULE_BUILD_FUNCS.registe_with_name(module_name="hdino")
def build_hdino(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share

    model = HDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        aux_query_number=args.aux_query_number,
        aux_box_noise_scale=args.aux_box_noise_scale,
        text_encoder_type=args.text_encoder_type,
        max_labels=args.max_labels,
        max_text_len=args.max_text_len,
    )

    return model




