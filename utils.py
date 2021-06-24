import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf


def display(img):
    dpi = mpl.rcParams['figure.dpi']
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    dims = img.shape
    h, w = dims[0], dims[1]
    figsize = w / float(dpi), h / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img, cmap='gray')
    plt.show()

def draw_boxes(img, bbox, labels):
    final_img = np.ascontiguousarray(img)
    for box, label in zip(bbox, labels):
        y1, x1, y2, x2 = box
        img_height = img.shape[0]
        color = tuple([np.random.randint(0,255) for i in range(3)])
        thickness = 2
        img = cv2.rectangle(img, (y1, x1), (y2, x2), color, thickness)
        color = (255,255,255)
        img = cv2.putText(img, label, (y1, x1-12), 0, 1e-3 * img_height, color, thickness)
    return img


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = tf.math.maximum(box1_x1, box2_x1)
    y1 = tf.math.maximum(box1_y1, box2_y1)
    x2 = tf.math.minimum(box1_x2, box2_x2)
    y2 = tf.math.minimum(box1_y2, box2_y2)

    intersection = tf.clip_by_value(x2-x1, 0.0, float('inf')) * \
                   tf.clip_by_value(y2-y1, 0.0, float('inf'))

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
