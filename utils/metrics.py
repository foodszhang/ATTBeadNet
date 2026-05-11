# from https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/metrics/stream_metrics.py

import numpy as np
import torch
import tqdm
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
# from sklearn.metrics import confusion_matrix


def extract_gt_centers_from_mask(mask: np.ndarray, mode: str = "pixel") -> list:
    """
    Extract center coordinates from binary mask.
    mode "pixel": each non-zero pixel is a center (for single-pixel annotations).
    mode "cc": connected component centroid.
    """
    centers = []
    K = mask.shape[0]

    for k in range(K):
        mask_k = mask[k]
        if isinstance(mask_k, torch.Tensor):
            mask_k = mask_k.cpu().numpy()
        binary = (mask_k > 0).astype(np.uint8)

        if mode == "pixel":
            ys, xs = np.nonzero(binary)
            for y, x in zip(ys, xs):
                centers.append({"class_id": k, "y": float(y), "x": float(x)})
        elif mode == "cc":
            from skimage import measure
            labeled = measure.label(binary, connectivity=1)
            props = measure.regionprops(labeled)
            for prop in props:
                y, x = prop.centroid
                centers.append({"class_id": k, "y": float(y), "x": float(x)})

    return centers


def extract_pred_centers_from_prob(
    probs: np.ndarray,
    thresholds: list,
    min_distances: list,
    use_peak_local_max: bool = True
) -> list:
    """
    Extract center coordinates from probability maps.
    Uses peak_local_max when use_peak_local_max=True (respects min_distance).
    Falls back to connected component centroid otherwise.
    """
    centers = []
    K = probs.shape[0]

    for k in range(K):
        prob_k = probs[k]
        thresh = thresholds[k] if k < len(thresholds) else 0.5
        min_d = min_distances[k] if k < len(min_distances) else 3

        if use_peak_local_max:
            coords = peak_local_max(
                prob_k,
                min_distance=min_d,
                threshold_abs=thresh,
                exclude_border=False,
            )
            for y, x in coords:
                centers.append({
                    "class_id": k,
                    "y": float(y),
                    "x": float(x),
                    "score": float(prob_k[y, x]),
                })
        else:
            from skimage import measure
            binary = prob_k > thresh
            labeled = measure.label(binary, connectivity=2)
            props = measure.regionprops(labeled, intensity_image=prob_k)
            for prop in props:
                centers.append({
                    "class_id": k,
                    "y": prop.centroid[0],
                    "x": prop.centroid[1],
                    "score": prop.mean_intensity,
                })

    return centers


def compute_distance_matrix(pred_centers: list, gt_centers: list, match_radius: float) -> np.ndarray:
    """Build cost matrix for Hungarian matching."""
    if len(pred_centers) == 0 or len(gt_centers) == 0:
        return np.full((len(pred_centers), len(gt_centers)), 1e9)

    cost = np.full((len(pred_centers), len(gt_centers)), 1e9, dtype=np.float32)

    for i, pred in enumerate(pred_centers):
        for j, gt in enumerate(gt_centers):
            if pred['class_id'] == gt['class_id']:
                dist = np.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)
                cost[i, j] = dist

    return cost


def hungarian_match(pred_centers: list, gt_centers: list, match_radius: float) -> tuple:
    """
    Match predictions to GT using Hungarian algorithm within match_radius.
    Returns: (matched, unmatched_pred, unmatched_gt)
    """
    cost = compute_distance_matrix(pred_centers, gt_centers, match_radius)

    if len(pred_centers) == 0:
        return [], [], list(range(len(gt_centers)))
    if len(gt_centers) == 0:
        return [], list(range(len(pred_centers))), []

    row_indices, col_indices = linear_sum_assignment(cost)

    matched = []
    unmatched_pred = set(range(len(pred_centers)))
    unmatched_gt = set(range(len(gt_centers)))

    for r, c in zip(row_indices, col_indices):
        if cost[r, c] <= match_radius:
            matched.append((r, c))
            unmatched_pred.discard(r)
            unmatched_gt.discard(c)

    return matched, list(unmatched_pred), list(unmatched_gt)

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

def validate(model, loader, device, metrics: StreamSegMetrics):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

        score = metrics.get_results()
    return score, ret_samples

# class AverageMeter(object):
#     """Computes average values"""
#     def __init__(self):
#         self.book = dict()

#     def reset_all(self):
#         self.book.clear()
    
#     def reset(self, id):
#         item = self.book.get(id, None)
#         if item is not None:
#             item[0] = 0
#             item[1] = 0

#     def update(self, id, val):
#         record = self.book.get(id, None)
#         if record is None:
#             self.book[id] = [val, 1]
#         else:
#             record[0]+=val
#             record[1]+=1

#     def get_results(self, id):
#         record = self.book.get(id, None)
#         assert record is not None
#         return record[0] / record[1]
