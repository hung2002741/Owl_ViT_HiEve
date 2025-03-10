from model import *
from data import *
from losses import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pprint
from PIL import Image
from transformers import OwlViTProcessor
import json
import os
import shutil
import yaml
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.io import write_png
from model import PostProcess

from train_util import (
    coco_to_model_input,
    labels_to_classnames,
    model_output_to_image,
    update_metrics
)
from util import BoxUtil, GeneralLossAccumulator, ProgressFormatter

from transformers import OwlViTProcessor


def get_training_config():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)
        return data["training"]

# Define a directory to save checkpoints
checkpoint_dir = "checkpoints"

# Create the directory if it doesn't exist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = MeanAveragePrecision(box_format = "cxcywh",iou_type="bbox", class_metrics=True).to(device)
    scaler = torch.cuda.amp.GradScaler()
    general_loss = GeneralLossAccumulator()
    progress_summary = ProgressFormatter()

    if os.path.exists("debug"):
        shutil.rmtree("debug")
    training_cfg = get_training_config()
    
    
    train_dataloader, test_dataloader, scales, labelmap = get_dataloaders()

    model = OwlViTForObjectDetectionModel.from_pretrained("google/owlvit-base-patch32", ignore_mismatched_sizes= True)
    for param in model.owlvit.parameters():
        param.requires_grad = False

    model.to(device)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

    # print([n for n, p in model.named_parameters() if p.requires_grad])
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    class_loss_coef, bbox_loss_coef, giou_loss_coef = training_cfg["class_loss_coef"], training_cfg["bbox_loss_coef"], training_cfg["giou_loss_coef"]

    criterion = Loss(scales= None,class_loss_coef= class_loss_coef, bbox_loss_coef=bbox_loss_coef, giou_loss_coef=giou_loss_coef)

    postprocess = PostProcess(
        confidence_threshold=training_cfg["confidence_threshold"],
        iou_threshold=training_cfg["iou_threshold"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=training_cfg["weight_decay"],
    )
    model.train()
    classMAPs = {v: [] for v in list(labelmap.values())}

    for epoch in range(training_cfg["n_epochs"]):
        model.train()
        if training_cfg["save_eval_images"]:
            os.makedirs(f"debug/{epoch}", exist_ok=True)

        # Train loop
        losses = []
        for i, (image, labels, boxes, text_queries, metadata) in enumerate(
            tqdm(train_dataloader, ncols=60)):
            # train_dataloader
            optimizer.zero_grad()
            
            # Prep inputs
            image = image.to(device)
            
            labels = labels.to(device)
            convert_text_queries = [item[0] for item in text_queries]
            # text_queries = text_queries.to(device)
            
            # Converting boxes from COCO format [xywh] to [cxcywh] normalize by image size
            boxes = coco_to_model_input(boxes, metadata).to(device)
            
            
            #inputs = processor(images = Image.open(metadata['impath'][0]).convert('RGB'), text= convert_text_queries, return_tensors="pt")
            inputs = processor(images = image, text= convert_text_queries, return_tensors= "pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            
            # Get predictions and save output
            pred_logits, pred_boxes = outputs['logits'], outputs['pred_boxes']
            losses = criterion(pred_logits, labels, pred_boxes, boxes)
            loss = (
                losses["loss_ce"]
                + losses["loss_bg"]
                + losses["loss_bbox"] 
                + losses["loss_giou"]
            )
            
            loss.backward()
            optimizer.step()

            general_loss.update(losses)  

        train_metrics = general_loss.get_values()
        general_loss.reset()

        # Eval loop
        model.eval()
        with torch.no_grad():
            for i, (image, labels, boxes, text_queries, metadata) in enumerate(
                tqdm(test_dataloader, ncols=60)):
    
                # Prep inputs
                image = image.to(device)
                labels = labels.to(device)
                convert_text_queries = [item[0] for item in text_queries]
                #text_queries = text_queries.to(device)
               
                # Converting boxes from COCO format [xywh] to [cxcywh] normalize by image size
                boxes = coco_to_model_input(boxes, metadata).to(device)
                
                #inputs = processor(images =Image.open(metadata['impath'][0]).convert('RGB'), text= convert_text_queries, return_tensors="pt")
                inputs = processor(images = image, text= convert_text_queries, return_tensors= "pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)

                # Get predictions and save output
                pred_logits, pred_boxes = outputs['logits'], outputs['pred_boxes']
                # Normalize pred_logits before postprocessing
                pred_logits_normalized = pred_logits.softmax(-1)
                pred_boxes, pred_classes, scores = postprocess(pred_boxes, pred_logits_normalized)
                
                update_metrics(metric,
                               metadata,
                               pred_boxes,
                               pred_classes, 
                               scores,
                               boxes,
                               labels)
                if training_cfg["save_eval_images"]:
                    pred_classes_with_names = labels_to_classnames(
                        pred_classes, labelmap
                    )
                    pred_boxes = model_output_to_image(pred_boxes.cpu(), metadata)
                    image_with_boxes = BoxUtil.draw_box_on_image(
                        metadata["impath"].pop(),
                        pred_boxes,
                        pred_classes_with_names,
                    )

                    write_png(image_with_boxes, f"debug/{epoch}/{i}.jpg")

        print("Computing metrics...")
        
        val_metrics = metric.compute()

        # for i, p in enumerate(val_metrics["map_per_class"].tolist()):
        #     label = labelmap[str(i)]
        #     classMAPs[label].append(p)

        # with open("class_maps.json", "w") as f:
        #     json.dump(classMAPs, f)

        # # Save a checkpoint at the end of each epoch in the checkpoint directory
        # checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        # checkpoint = {
        #     "epoch": epoch,
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        #     "train_metrics": train_metrics,
        #     "val_metrics": val_metrics,
        # }
        # torch.save(checkpoint, checkpoint_path)
        # print("Checkpoints saved")
        # Handle map_per_class as a scalar for class "person" (label 1)

        map_value = val_metrics["map_per_class"]
        if isinstance(map_value, torch.Tensor):
            map_value = map_value.item()  # Convert tensor to scalar if needed
        print(f"mAP for 'person': {map_value}")

        # Update classMAPs for "person" (label 1)
        classMAPs["person"].append(map_value)  # Directly append to "person" key

        with open("class_maps.json", "w") as f:
            json.dump(classMAPs, f)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, checkpoint_path)
        print("Checkpoints saved")

        metric.reset()
        progress_summary.update(epoch, train_metrics, val_metrics)
        progress_summary.print()







