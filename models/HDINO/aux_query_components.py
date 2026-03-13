# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]

# Modified from [DINO]


import torch
from .misc import inverse_sigmoid
from .box_ops import box_iou_one2one


def prepare_for_aux_query(aux_query_args, training, num_queries, hidden_dim, one2many_embeddings, device):
    if training:
        targets, aux_query_number, box_noise_scale = aux_query_args
        aux_query_number = aux_query_number * 2

        known = []
        labels = []
        boxes = []
        batch_idx = []

        for i, t in enumerate(targets):
            label = t['labels']
            bbox = t['boxes']
            assert len(bbox)>0, "all training samples should have gt"
            known.append(torch.ones_like(label, device=device))
            labels.append(label)
            boxes.append(bbox)
            batch_idx.append(torch.full_like(label.long(), i))


        labels = torch.cat(labels)
        boxes = torch.cat(boxes)
        batch_idx = torch.cat(batch_idx)

        batch_size = len(known)
        batch_num = 0
        known_num = []
        for k in known:
            known_num.append(sum(k))
            batch_num += sum(k)

        if int(max(known_num)) == 0:
            aux_query_number = 1
        else:
            if aux_query_number >= 100:
                aux_query_number = aux_query_number // (int(max(known_num) * 2))
            elif aux_query_number < 1:
                aux_query_number = 1
        if aux_query_number == 0:
            aux_query_number = 1

        known_labels = labels.repeat(2 * aux_query_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * aux_query_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * aux_query_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * aux_query_number)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2  # x1y1
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2  # x2y2

            known_bbox_xyxy = known_bbox_.clone()

            num = batch_num * 2 if aux_query_number ==1 else batch_num * 4
            known_bboxs_1 = known_bbox_[:-num, :]
            known_bboxs_2 = known_bbox_[-num:, :]

            diff_1 = torch.zeros_like(known_bboxs_1)
            diff_1[:, :2] = known_bboxs_1[:, 2:] / 2
            diff_1[:, 2:] = known_bboxs_1[:, 2:] / 2

            rand_sign_1 = torch.randint_like(known_bboxs_1, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part_1 = torch.rand_like(known_bboxs_1)
            rand_part_1 *= rand_sign_1
            known_bboxs_1 = known_bboxs_1 + torch.mul(rand_part_1,diff_1).to(device) * box_noise_scale
            known_bboxs_1 = known_bboxs_1.clamp(min=0.0, max=1.0)

            diff_2 = torch.zeros_like(known_bboxs_2)
            diff_2[:, :2] = known_bboxs_2[:, 2:] / 2
            diff_2[:, 2:] = known_bboxs_2[:, 2:] / 2
            rand_sign_2 = torch.tensor([[-1, -1, 1, 1]], dtype=torch.float32, device=device).repeat(known_bboxs_2.shape[0], 1)
            rand_part_2 = torch.rand(size=(known_bboxs_2.shape[0], 2), dtype=torch.float32, device=device).repeat(1,2)
            rand_part_2 *= rand_sign_2
            known_bboxs_2 = known_bboxs_2 + torch.mul(rand_part_2,diff_2).to(device) * box_noise_scale
            known_bboxs_2 = known_bboxs_2.clamp(min=0.0, max=1.0)

            known_bbox_ = torch.cat([known_bboxs_1, known_bboxs_2], dim=0)

            known_noise_bbox_xyxy = known_bbox_.clone()

            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]


        input_label_embed = []
        all_iou = []
        for i in range(2 * aux_query_number):
            input_label_embed_i = [one2many_embeddings.weight[num * i:num * (i + 1)] for num in known_num]
            input_label_embed.extend(input_label_embed_i)

            iou, _ = box_iou_one2one(known_bbox_xyxy[:batch_num, :], known_noise_bbox_xyxy[i * batch_num:(i + 1) * batch_num, :])
            all_iou.append(iou.clamp(0, 1))
        all_iou = torch.cat(all_iou)

        input_label_embed = torch.cat(input_label_embed, dim=0)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim, device=device)
        padding_bbox = torch.zeros(pad_size, 4, device=device)
        padding_iou = torch.ones(pad_size, device=device, dtype=torch.float32)

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)
        input_iou = padding_iou.repeat(batch_size, 1)

        map_known_indice = torch.tensor([]).to(device)
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * aux_query_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed
            input_iou[(known_bid.long(), map_known_indice)] = all_iou

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size, device=device) < 0
        attn_mask[pad_size:, :pad_size] = True
        aux_query_meta = {
            'pad_size': pad_size,
            'num_aux_query_group': aux_query_number,
            'iou': input_iou,
            'known_bid':known_bid.long(),
            'map_known_indice':map_known_indice,
            'single_pad':single_pad,
            'labels': labels
        }
    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        aux_query_meta = None

    return input_query_label, input_query_bbox, attn_mask, aux_query_meta

def aux_query_post_process(outputs_class, outputs_coord, aux_query_meta, aux_loss, _set_aux_loss):
    if aux_query_meta and aux_query_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :aux_query_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :aux_query_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, aux_query_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, aux_query_meta['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord)
        aux_query_meta['output_known_lbs_bboxes'] = out
    return outputs_class, outputs_coord


