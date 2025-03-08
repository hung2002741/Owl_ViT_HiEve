import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_area, box_convert
import numpy as np

from matcher import HungarianMatcher, box_iou, generalized_box_iou

class Loss(torch.nn.Module):
    def __init__(self, scales, class_loss_coef, bbox_loss_coef, giou_loss_coef):
        super().__init__()
        self.matcher = HungarianMatcher(1)  # Single class (person)
        self.class_criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")  # Binary cross-entropy
        self.background_label = 0  # 0 for background, 1 for person
        self.class_loss_coef = class_loss_coef
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef

    def class_loss(self, outputs, target_classes, class_loss_coef):
        """
        Binary classification loss for person vs. background.
        """
        src_logits = outputs["pred_logits"]  # Shape: [batch_size, num_boxes, 1]
        src_logits = src_logits.squeeze(0)  # Shape: [num_boxes, 1]
        target_classes = target_classes.squeeze(0)  # Shape: [num_boxes]

        # Split into positive (person) and negative (background)
        pos_mask = target_classes == 1  # Person class
        neg_mask = target_classes == 0  # Background

        pred_logits_pos = src_logits[pos_mask]  # Predictions for person boxes
        pred_logits_neg = src_logits[neg_mask]  # Predictions for background boxes

        # Targets: 1 for person, 0 for background
        pos_targets = torch.ones_like(pred_logits_pos)  # Shape: [num_pos, 1]
        neg_targets = torch.zeros_like(pred_logits_neg)  # Shape: [num_neg, 1]

        # Compute loss
        if pred_logits_pos.numel() > 0:
            pos_loss = self.class_criterion(pred_logits_pos, pos_targets)
        else:
            pos_loss = torch.tensor(0.0, device=src_logits.device)

        if pred_logits_neg.numel() > 0:
            neg_loss = self.class_criterion(pred_logits_neg, neg_targets)
        else:
            neg_loss = torch.tensor(0.0, device=src_logits.device)

        return pos_loss * class_loss_coef, neg_loss * class_loss_coef

    def loss_boxes(self, outputs, targets, indices, idx, num_boxes, bbox_loss_coef, giou_loss_coef):
        assert "pred_boxes" in outputs
        src_boxes = outputs["pred_boxes"][idx]  # Shape: [num_boxes, 4]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = torch.nn.functional.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_convert(target_boxes, in_fmt='cxcywh', out_fmt='xyxy'),
            box_convert(src_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        ))
        loss_giou = loss_giou.sum() / num_boxes

        return loss_bbox * bbox_loss_coef, loss_giou * giou_loss_coef

    def forward(self, predicted_classes, target_classes, predicted_boxes, target_boxes):
        # Format inputs
        in_preds = {
            "pred_logits": predicted_classes,  # Shape: [batch_size, num_boxes, 1]
            "pred_boxes": predicted_boxes,     # Shape: [batch_size, num_boxes, 4]
        }
        in_targets = [
            {"labels": _labels, "boxes": _boxes}
            for _boxes, _labels in zip(target_boxes, target_classes)
        ]

        # Match predictions to targets
        target_classes, indices, idx = self.matcher(in_preds, in_targets)

        # Compute box losses
        num_boxes = sum(len(t["labels"]) for t in in_targets)
        loss_bbox, loss_giou = self.loss_boxes(
            in_preds, in_targets, indices, idx, num_boxes,
            bbox_loss_coef=self.bbox_loss_coef, giou_loss_coef=self.giou_loss_coef
        )

        # Compute class loss
        loss_class, loss_background = self.class_loss(in_preds, target_classes, class_loss_coef=self.class_loss_coef)

        losses = {
            "loss_ce": loss_class,
            "loss_bg": loss_background,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
        return losses