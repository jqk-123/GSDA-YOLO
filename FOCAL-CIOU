iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
 d = 0.00
 u = 0.95
 if torch.all(iou > u):
    iou = 1
elif torch.all(iou < d):
    iou = 0
else:
 iou = ((iou - d) / (u - d)).clamp(0, 1)
loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
