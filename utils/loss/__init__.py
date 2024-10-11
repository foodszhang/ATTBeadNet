
from .u3ploss import build_u3p_loss
import torch
import torch.nn as nn


def get_loss(loss_function):
    """Get optimization criterion / loss function.

    :param loss_function: Wanted optimization criterion / loss function.
        :type loss_function: str
    :return: Optimization criterion / loss function.
    """

    if loss_function == 'bce_dice':

        criterion = bce_dice

    elif loss_function == 'bce':

        criterion = nn.BCEWithLogitsLoss()

    elif loss_function == 'ce':

        criterion = nn.CrossEntropyLoss()

    elif loss_function == 'dice':

        criterion = dice_loss

    else:

        raise Exception('Loss function "{}" not known!'.format(loss_function))

    return criterion


def dice_loss(y_pred, y_true):
    """Dice loss: harmonic mean of precision and recall (FPs and FNs are weighted equally). Only for 1 output channel.

    :param y_pred: Prediction [batch size, channels, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, height, width].
        :type y_true:
    :return:
    """

    # Avoid division by zero
    smooth = 1.

    # Apply sigmoid activation to prediction
    pred = torch.sigmoid(y_pred)

    # Flatten predition and ground truth
    gt = y_true.contiguous().view(-1)
    pred = pred.contiguous().view(-1)

    # Calculate Dice loss
    intersection = torch.sum(gt * pred)
    loss = 1 - (2. * intersection + smooth) / (torch.sum(gt**2) + torch.sum(pred**2) + smooth)
    
    return loss


def bce_dice(y_pred, y_true):
    """ Sum of binary cross-entropy loss and Dice loss. Only for 1 output channel.

    :param y_pred: Prediction [batch size, channels, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, height, width].
        :type y_true:
    :return:
    """
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(y_pred, y_true) + 0.5 * dice_loss(y_pred, y_true)
    
    return loss

