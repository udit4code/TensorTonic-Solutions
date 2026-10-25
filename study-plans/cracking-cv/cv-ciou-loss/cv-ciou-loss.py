import math
import numpy as np 

def get_ciou_loss_per_box(pred_box: list, target_box: list) -> float:
    """
    Calculates the CIoU loss for a single pair of prediction and target boxes
    using the exact formulation from the reference solution.
    """
    px1, py1, px2, py2 = pred_box
    gx1, gy1, gx2, gy2 = target_box
    
    # Step 1 : Get Intersection & Union
    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    
    area_p = (px2 - px1) * (py2 - py1)
    area_g = (gx2 - gx1) * (gy2 - gy1)
    union = area_p + area_g - inter
    
    eps = 1e-7
    iou = inter / (union + eps)
    
    # Step 2 : Get Distance Penalty (DIoU terms)
    cx_p = (px1 + px2) / 2.0
    cy_p = (py1 + py2) / 2.0
    cx_g = (gx1 + gx2) / 2.0
    cy_g = (gy1 + gy2) / 2.0
    rho2 = (cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2
    
    # Step 3 : Get Enclosing Diagonal Squared
    cx1 = min(px1, gx1)
    cy1 = min(py1, gy1)
    cx2 = max(px2, gx2)
    cy2 = max(py2, gy2)
    c_diag2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + eps
    
    # Step 4 : Get Aspect Ratio Penalty
    w_p = px2 - px1
    h_p = py2 - py1
    w_g = gx2 - gx1
    h_g = gy2 - gy1
    
    # Step 5 : Added safe division to prevent ZeroDivisionError if box height is 0
    h_p_safe = h_p if h_p != 0.0 else eps
    h_g_safe = h_g if h_g != 0.0 else eps
    
    v = (4.0 / (math.pi ** 2)) * (math.atan(w_g / h_g_safe) - math.atan(w_p / h_p_safe)) ** 2
    alpha = v / (1.0 - iou + v + eps)
    
    # Step 6 : Get Final CIoU Loss calculation
    diou_loss = 1.0 - iou + (rho2 / c_diag2)
    ciou_loss_val = diou_loss + (alpha * v)
    
    return float(ciou_loss_val)
    
def ciou_loss(pred_boxes: list, target_boxes: list) -> float:
    """
    Returns the mean CIoU loss as a float, rounded to 4 decimals.
    """
    assert len(pred_boxes) == len(target_boxes)
    n = len(pred_boxes)
    if n == 0:
        return 0.0
        
    total_loss = 0.0
    for p, g in zip(pred_boxes, target_boxes):
        total_loss += get_ciou_loss_per_box(p, g)
        
    mean_loss = total_loss / n
    return round(float(mean_loss), 4)