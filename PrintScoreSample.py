import kp
import cv2
import os
import argparse
import datetime

import warnings
warnings.filterwarnings('ignore')

import sys
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))

import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from tqdm import tqdm

###
# run resizer before running evaluation

###
dataset_dir1 = r"/home/alarm/kneopi-examples/ai_application/plus_python/kneron_test_dataset/"

###
dataset_dir = r"/home/alarm/kneopi-examples/ai_application/plus_python/test_dataset/"
image_dir = os.path.join(dataset_dir, "images/")
label_dir = os.path.join(dataset_dir, "labels/")

dateStr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
result_dir_path = f"/home/alarm/kneopi-examples/ai_application/plus_python/evaluation_result_{dateStr}"
nowstr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# evaluation metric (mAP)
metric = MeanAveragePrecision(iou_type="bbox")

# target class
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
target_classes = [0] # person only

#
all_preds = []
all_targets = []
###

###
ANCHORS = [[(12, 16), (19, 36), (40, 28)],
           [(36, 75), (76, 55), (72, 146)],
           [(142, 110), (192, 243), (459, 401)]]

STRIDES = [8, 16, 32]

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']

rng = np.random.default_rng(3)
colors = (rng.uniform(0, 255, size=(len(class_names), 3))).astype(np.uint8)  # different colors for different classes


IMAGE_FILE_PATH = None
MODEL_FILE_PATH = None
SCPU_FW_PATH = None

_generic_inference_input_descriptor = kp.GenericImageInferenceDescriptor()
###


# utils
def draw_detections(img_path, boxes, scores, class_ids, pad_lt, scale, mask_alpha=0.3):
    """
    image: PIL.Image (RGB)
    boxes: Nx4
    scores: N
    class_ids: N
    """

    # bbox xyxy 좌표 복원 (원 이미지)
    boxes[:, 0] = (boxes[:, 0] - pad_lt[0]) / scale
    boxes[:, 1] = (boxes[:, 1] - pad_lt[1]) / scale
    boxes[:, 2] = (boxes[:, 2] - pad_lt[0]) / scale
    boxes[:, 3] = (boxes[:, 3] - pad_lt[1]) / scale

    # PIL 이미지 복제
    # det_img = image.copy()
    det_img = Image.open(img_path).convert("RGB")
    mask_img = Image.new("RGB", det_img.size, (0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_img)
    det_draw = ImageDraw.Draw(det_img)

    img_w, img_h = det_img.size

    font = ImageFont.load_default()

    for b, s, c in zip(boxes, scores, class_ids):
        color = tuple(int(v) for v in colors[c])   # 이미 BGR이 아닌 RGB로 생성됨
        x1, y1, x2, y2 = map(int, b)

        # ───────────── Box 그리기 (det_img)
        det_draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # ───────────── Mask (반투명) 박스 (mask_img)
        mask_draw.rectangle([x1, y1, x2, y2], fill=color)

        # ───────────── Label
        label = f"{class_names[c]} {int(s * 100)}%"

        # 텍스트 크기
        # Pillow ≥10.0에서는 textbbox 사용
        bbox = det_draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # 배경 박스 (det_img)
        det_draw.rectangle([x1, y1 - th, x1 + tw, y1], fill=color)

        # 텍스트
        det_draw.text((x1, y1 - th), label, fill=(255, 255, 255), font=font)

    # ───────────── 투명도 합성 (OpenCV addWeighted 대체)
    blended = Image.blend(det_img, mask_img, mask_alpha)
    final = Image.blend(blended, det_img, 1 - mask_alpha)

    return final


# preprocess
def letterbox(img: Image.Image, new_shape=(640, 640), color=(114, 114, 114)):
    """
    PIL 이미지를 비율 유지하며 resize + padding(letterbox) 처리하는 함수
    img: PIL.Image
    new_shape: (width, height)
    color: 패딩 색상 (default: YOLO 스타일 114)
    """
    org_w, org_h = img.size
    target_w, target_h = new_shape

    # 비율 유지한 resize 크기 계산 (원본 대 640 비율)
    scale = min(target_w / org_w, target_h / org_h)
    new_w = int(org_w * scale)
    new_h = int(org_h * scale)

    # scale 대로 이미지 resize
    resized_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # padding 계산
    ttl_pad_w = target_w - new_w
    ttl_pad_h = target_h - new_h

    left_pad = ttl_pad_w // 2
    right_pad = ttl_pad_w - left_pad

    top_pad = ttl_pad_h // 2
    bottom_pad = ttl_pad_h - top_pad

    # resize 한 이미지에 padding 추가
    padded_img = ImageOps.expand(resized_img, border=(left_pad, top_pad, right_pad, bottom_pad), fill=color)

    return padded_img, (left_pad, top_pad), scale


def preprocess(img_path, model_input=(640, 640)):
    """
     PIL 기반 YOLO preprocess
     - 최종 shape 정확히 (1,3,H,W)
     """
    # read image (RGB)
    img = Image.open(img_path).convert("RGB")

    # resize & add paddings (letterbox)
    img, pad_lt, scale = letterbox(img, model_input)

    # convert to np array & normalize
    img_np = np.array(img).astype(np.float32) / 256.0 - 0.5

    # (HWC) -> (CHW + batch dimension)
    img_np = np.transpose(img_np, (2, 0, 1))[None, ...]

    return img_np, pad_lt, scale


# postprocess
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    interWidth = max(0, x2 - x1)
    interHeight = max(0, y2 - y1)
    interArea = interWidth * interHeight
    if interArea == 0:
        return 0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    totalArea = area1 + area2 - interArea
    iou = interArea / totalArea

    return iou


def nms(boxes, scores, iou_thres=0.45):
    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        cur = idxs[0]
        keep.append(cur)
        if len(idxs) == 1:
            break

        rest = idxs[1:]

        suppressed = []
        for i in rest:
            if iou(boxes[cur], boxes[i]) > iou_thres:
                suppressed.append(i)

        idxs = np.array([i for i in rest if i not in suppressed])

    return keep


def postprocess(boxes, scores, class_ids, iou_thres=0.45):
    # class wise NMS
    final_boxes = []
    final_scores = []
    final_cls = []

    if len(boxes) > 0:
        for c in np.unique(class_ids):
            idxs = np.where(class_ids == c)[0]
            keep = nms(boxes[idxs], scores[idxs], iou_thres)
            for k in keep:
                final_boxes.append(boxes[idxs][k])
                final_scores.append(scores[idxs][k])
                final_cls.append(c)

    # cast to np array
    boxes = np.array(final_boxes)
    scores = np.array(final_scores)
    class_ids = np.array(final_cls)

    return boxes, scores, class_ids


# output decoding
def to_5d(arr, num_anchors=3):
    # Kneron 출력 객체라면 실제 NumPy 배열을 가져옴
    if hasattr(arr, "ndarray"):
        arr = arr.ndarray

    arr = np.asarray(arr)
    ##

    N, C, H, W = arr.shape  # (1, 33, 80, 80)
    no = C // num_anchors  # per-anchor len  -> no = 11
    arr = arr.reshape(N, num_anchors, no, H, W)  # (1, 3, 11, 80, 80)
    arr = arr.transpose(0, 1, 3, 4, 2)  # (1,A,H,W,no)    -> (1, 3, 80, 80, 11)
    return arr


def decode_onnx_outputs(outputs, conf_thres):
    all_boxes = []
    all_scores = []
    all_classes = []

    for i, arr in enumerate(outputs):                     # 3 output channels (corresp. to 3 anchors)
        arr = to_5d(arr)                                  # reshape the data -> arr: (1,C,H,W) → (1,A,H,W,no)
        _, A, H, W, no = arr.shape

        anchor_set = ANCHORS[i]                           # i: anchor & stride idx (select corresp anchor & stride
        stride = STRIDES[i]

        for a in range(A):                                # A = 3  - inner anchor tuple idx
            aw, ah = anchor_set[a]

            for y in range(H):                            # H = 80, 40, 20
                for x in range(W):                        # W = 80, 40, 20
                    px, py, pw, ph = arr[0, a, y, x, :4]  # 0~3  bbox xywh
                    obj = arr[0, a, y, x, 4]              # 4    obj confidence

                    if obj < conf_thres:                  # confidence filtering
                        continue

                    class_scores = arr[0, a, y, x, 5:]    # 5~10 (cls probs)
                    cls_id = np.argmax(class_scores)      # idx of the highest class prob
                    cls_score = class_scores[cls_id]      # highest class prob

                    conf = obj * cls_score                # obj conf * class prob
                    if conf < conf_thres:                 # 2nd filtering
                        continue

                    bx = (px * 2 - 0.5 + x) * stride      # bbox center x   (yolov7's anchor based bbox restoration
                    by = (py * 2 - 0.5 + y) * stride      # bbox center y   (bbox 가 grid cell 주변으로 크게 벗어나지 않도록 하기 위해 다음과 같이 변환하여 제한된 범위에서 remapping)  -0.5~1.5 범위로 확장 (px * 2 - 0.5), 현재 grid cell의 정수 위치 추가 (+x), covert 'feature map coord' to 'actual input img coord(*stride)
                    bw = (pw * 2) ** 2 * aw               # bbox width      (0~4 범위 값으로 확장 )(pw * w)**w),  현재 feature map 에 해당하는 anchor width (*aw) )
                    bh = (ph * 2) ** 2 * ah               # bbox height

                    x1 = bx - bw / 2                      # xywh -> x1y1x2y2
                    y1 = by - bh / 2
                    x2 = bx + bw / 2
                    y2 = by + bh / 2

                    all_boxes.append([x1, y1, x2, y2])    # bbox xxyy  (append decoded results
                    all_scores.append(conf)               # score
                    all_classes.append(cls_id)            # cls

    if len(all_boxes) == 0:
        return [], [], []

    all_boxes = np.array(all_boxes)                       # cast to numpy array
    all_scores = np.array(all_scores)
    all_classes = np.array(all_classes)

    return all_boxes, all_scores, all_classes


# evaluation
def calculate_tp_fp_fn(pred_boxes, pred_scores, pred_class_ids,
                       gt_boxes, gt_class_ids, iou_threshold=0.5):
    """
    클래스 일치 + IoU threshold 기준으로 일대일 매칭.
    같은 정답 객체는 한 번만 TP로 계산한다.
    """
    pred_boxes = torch.as_tensor(
        np.asarray(pred_boxes).reshape(-1, 4), dtype=torch.float32
    )
    gt_boxes = torch.as_tensor(
        np.asarray(gt_boxes).reshape(-1, 4), dtype=torch.float32
    )
    pred_scores = np.asarray(pred_scores).reshape(-1)
    pred_class_ids = np.asarray(pred_class_ids).reshape(-1)
    gt_class_ids = np.asarray(gt_class_ids).reshape(-1)

    num_preds = len(pred_boxes)
    num_gt = len(gt_boxes)

    if num_preds == 0:
        return 0, 0, num_gt

    if num_gt == 0:
        return 0, num_preds, 0

    ious = box_iou(pred_boxes, gt_boxes).cpu().numpy()

    # 높은 confidence 예측부터 매칭
    order = np.argsort(-pred_scores)
    matched_gt = set()
    tp = 0
    fp = 0

    for pred_idx in order:
        # 클래스가 같은 GT만 후보로 사용
        candidates = np.where(
            gt_class_ids == pred_class_ids[pred_idx]
        )[0]

        if len(candidates) == 0:
            fp += 1
            continue

        # 이미 다른 예측과 매칭된 GT는 제외
        candidates = [
            gt_idx for gt_idx in candidates
            if gt_idx not in matched_gt
        ]

        if not candidates:
            fp += 1
            continue

        best_gt = max(candidates, key=lambda j: ious[pred_idx, j])

        if ious[pred_idx, best_gt] >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = num_gt - len(matched_gt)
    return tp, fp, fn


def single_image_inference_test():
    ###
    # prepare generic image inference input descriptor
    _generic_inference_input_descriptor = kp.GenericImageInferenceDescriptor(
        model_id=model_nef_descriptor.models[0].id,
        inference_number=0,
    )

    # read image
    im0 = cv2.imread(IMAGE_FILE_PATH, cv2.IMREAD_COLOR)
    cvt_image = cv2.cvtColor(src=im0, code=cv2.COLOR_BGR2BGR565)
    inference_image = cvt_image

    print(inference_image.shape)

    #
    # 너비가 홀수이면 마지막 열 제거
    if inference_image.shape[1] % 2 != 0:
        inference_image = inference_image[:, :-1, :]

    # 높이가 홀수이면 마지막 행 제거
    if inference_image.shape[0] % 2 != 0:
        inference_image = inference_image[:-1, :, :]

    print("fixed img:", inference_image.shape)

    # preprocess version
    #preprocessed_img, pad_lt, scale = preprocess(IMAGE_FILE_PATH)
    #print(preprocessed_img.shape)
    #print(pad_lt)
    #print(scale)

    height, width = inference_image.shape[:2]
    scale = min(640 / width, 640 / height)
    print(scale)

    # descriptor
    _generic_inference_input_descriptor.input_node_image_list = [
        kp.GenericInputNodeImage(
            image=inference_image,
            #resize_mode=kp.ResizeMode.KP_RESIZE_ENABLE,
            #padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
            #normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON
        )
    ]

    #
    # starting inference
    print('[Starting Inference]')
    try:
        kp.inference.generic_image_inference_send(device_group=device_group,
                                                  generic_inference_input_descriptor=_generic_inference_input_descriptor)
        generic_raw_result = kp.inference.generic_image_inference_receive(device_group=device_group)
    except kp.ApiKPException as exception:
        print(' - Error: inference failed, error = {}'.format(exception))
        exit(0)

    # retrieve inference node output
    print('[Retrieve Inference Node Output ]')
    inf_node_output_list = []
    for node_idx in range(generic_raw_result.header.num_output_node):
        inference_float_node_output = kp.inference.generic_inference_retrieve_float_node(node_idx=node_idx,
                                                                                         generic_raw_result=generic_raw_result,
                                                                                         channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW)
        inf_node_output_list.append(inference_float_node_output)
    print(' - Success')

    #
    # Decode & Post Process
    print('[Decode & Post Process output]')

    print("inf_node_output_list:")
    print(len(inf_node_output_list))
    print(inf_node_output_list[0].shape)  # (1, 33, 80, 80)  (stride  8   8 * 80 = 640)
    print(inf_node_output_list[1].shape)  # (1, 33, 40, 40)  (       16  16 * 40 = 640)
    print(inf_node_output_list[2].shape)  # (1, 33, 20, 20)  (       32  32 * 20 = 640)

    conf_thres = 0.25
    iou_thres = 0.45

    boxes, scores, class_ids = decode_onnx_outputs(inf_node_output_list, conf_thres)

    print(len(boxes))
    print(len(scores))
    print(len(class_ids))

    # postprocess
    boxes, scores, class_ids = postprocess(boxes, scores, class_ids)

    print(len(boxes))
    print(len(scores))
    print(len(class_ids))

    # print results
    if len(boxes) > 0:
        print(boxes)
        print(scores)
        print(class_ids)
        final_img = draw_detections(IMAGE_FILE_PATH, boxes, scores, class_ids, (0, 0), scale)
        final_img.save("/home/alarm/kneopi-examples/ai_application/plus_python/rslt.png")


def evaluate_model():
    global _generic_inference_input_descriptor, all_targets, all_preds

    idx = -1
    # read image paths from current dir

    # count = 10
    # image_files = [filename for filename in os.listdir(image_dir)[:count] if filename.endswith((".jpg", ".png"))]

    image_paths = []
    label_paths = []

    for sub_dir in os.listdir(dataset_dir1):
        sub_path = os.path.join(dataset_dir1, sub_dir)

        img_dir = os.path.join(sub_path, "images/")
        lbl_dir = os.path.join(sub_path, "labels/")

        image_paths += [os.path.join(img_dir, path) for path in os.listdir(img_dir)]
        label_paths += [os.path.join(lbl_dir, path) for path in os.listdir(lbl_dir)]

    image_paths = sorted(image_paths)
    label_paths = sorted(label_paths)


    #image_files = [filename for filename in os.listdir(image_dir) if filename.endswith((".jpg", ".png"))]

    #for image_file in tqdm(image_files, desc="Evaluating"):
    for image_path in tqdm(image_paths, desc="Evaluating"):
        idx += 1

        ###
        # load an image
        #image_path = os.path.join(image_dir, image_file)

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        cvt_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2BGR565)
        inference_img = cvt_img

        h, w = inference_img.shape[:2]
        scale = min(640 / w, 640 / h)

        # error prevention
        # width is odd
        if inference_img.shape[1] % 2 != 0:
            inference_img = inference_img[:, :-1, :]
        # height is odd
        if inference_img.shape[0] % 2 != 0:
            inference_img = inference_img[:-1, :, :]

        # prepare generic image inference input descriptor
        _generic_inference_input_descriptor = kp.GenericImageInferenceDescriptor(
            model_id=model_nef_descriptor.models[0].id,
            inference_number=0,
        )

        _generic_inference_input_descriptor.input_node_image_list = [
            kp.GenericInputNodeImage(
                image=inference_img,
                # resize_mode=kp.ResizeMode.KP_RESIZE_ENABLE,
                # padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
                # normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON
            )
        ]
        ###

        ###
        # inference
        # print('[Starting Inference]')
        try:
            kp.inference.generic_image_inference_send(device_group=device_group,
                                                      generic_inference_input_descriptor=_generic_inference_input_descriptor)
            generic_raw_result = kp.inference.generic_image_inference_receive(device_group=device_group)
        except kp.ApiKPException as exception:
            print(' - Error: inference failed, error = {}'.format(exception))
            exit(0)

        # retrieve inference node output
        # print('[Retrieve Inference Node Output ]')
        inf_node_output_list = []
        for node_idx in range(generic_raw_result.header.num_output_node):
            inference_float_node_output = kp.inference.generic_inference_retrieve_float_node(node_idx=node_idx,
                                                                                             generic_raw_result=generic_raw_result,
                                                                                             channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW)
            inf_node_output_list.append(inference_float_node_output)
        # print(' - Success')

        # Decode & Post Process
        # print('[Decode & Post Process output]')

        conf_thres = 0.20
        iou_thres = 0.5

        decoded_boxes, decoded_scores, decoded_class_ids = decode_onnx_outputs(inf_node_output_list, conf_thres)
        pred_boxes, pred_scores, pred_class_ids = postprocess(decoded_boxes, decoded_scores, decoded_class_ids)

        # class filtering
        for id in target_classes:
            mask = pred_class_ids == id

            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_class_ids = pred_class_ids[mask]
        ###

        ###
        # load corresponding label
        #label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")
        label_path = label_paths[idx]
        if not os.path.exists(label_path):
            print("no_label")
            continue  # skip if there is no label

        # init boxes & labels
        boxes = []
        labels = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, cx, cy, bw, bh = map(float, parts)
                # print(parts)

                # normalize (into xywh - for labels in 0~1 float val)
                # cx = float(cx) * w
                # bw = float(bw) * w
                # cy = float(cy) * h
                # bh = float(bh) * h
                # boxes.append([cx, cy, bw, bh])

                # convert (into xyxy - for labels in 0~1 float val)
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(int(class_id))

        if not boxes:
            continue
        ###

        ###
        # label data
        targets = [{
            "boxes": torch.tensor(boxes, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }]

        # prediction data
        preds = [{
            "boxes": torch.tensor(pred_boxes, dtype=torch.float),
            # "boxes": result.boxes.xywh.cpu(),
            "scores": torch.tensor(pred_scores, dtype=torch.float),
            "labels": torch.tensor(pred_class_ids, dtype=torch.int64),
        }]

        # print pred & labels
        # print(targets)
        # print(preds)
        # print("=============")
        ###

        ###
        # eval- mAP metric
        metric.update(preds, targets)

        # match & eval- IoU, TP,FP,FN
        if len(targets[0]["boxes"]) > 0 and len(preds[0]["boxes"]) > 0:
            # matching & Iou evalulatons are done here
            # TP, FP, FN
            TP, FP, FN = calculate_tp_fp_fn(pred_boxes, pred_scores, pred_class_ids, boxes, labels, iou_threshold=iou_thres)
            #ious = box_iou(preds[0]["boxes"], targets[0]["boxes"])
            #max_ious, max_indices = ious.max(dim=1)

            ## eval TP,FP,FN
            #TP = (max_ious >= 0.5).sum()        # TP if IoU is over 0.5
            #FP = len(preds[0]["boxes"]) - TP
            #FN = len(targets[0]["boxes"]) - TP
        else:
            TP = 0
            FP = len(preds[0]["boxes"])
            FN = len(targets[0]["boxes"])

        # save data (for Precision/Recall/F1-score)
        all_preds.append(torch.tensor([TP, FP]))
        all_targets.append(torch.tensor([TP + FN]))

        # annotate & save results   <-
        if len(pred_boxes) > 0 and idx < 20:
            final_img = draw_detections(image_path, pred_boxes, pred_scores, pred_class_ids, (0, 0), scale)
            final_img.save(f"/home/alarm/kneopi-examples/ai_application/plus_python/result_imgs/rslt{idx}.png")
        ###


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser(description='KL730 Demo Generic Image Inference Example.')
    parser.add_argument('-p',
                        '--port_id',
                        help='Using specified port ID for connecting device (Default: port ID of first scanned Kneron '
                             'device)',
                        default=0,
                        type=int)
    parser.add_argument('-m',
                        '--model',
                        help='yolov7 nef model path',
                        default=os.path.join(PWD, '/home/alarm/models_730_rgb.nef'),
                        type=str)
    parser.add_argument('-img',
                        '--image_path',
                        help='test image path',
                        default=os.path.join(PWD, 'res/images/bus.jpg'),
                        type=str)
    parser.add_argument('-fw',
                        '--firmware_path',
                        help='firmware path',
                        default=os.path.join(PWD, 'res/firmware/KL730/kp_firmware.tar'),
                        type=str)
    args = parser.parse_args()

    # init
    usb_port_id = args.port_id
    IMAGE_FILE_PATH = args.image_path
    MODEL_FILE_PATH = args.model
    SCPU_FW_PATH = args.firmware_path

    # connect the device
    try:
        print('[Connecting Device]')
        device_group = kp.core.connect_devices(usb_port_ids=[usb_port_id])
        print(' - Success')
    except kp.ApiKPException as exception:
        print('Error: connect device fail, port ID = \'{}\', error msg: [{}]'.format(usb_port_id,
                                                                                     str(exception)))
        exit(0)

    # upload firmware to device
    try:
        print('[Uploading Firmware]')
        kp.core.load_firmware_from_file(device_group=device_group,
                                        scpu_fw_path=SCPU_FW_PATH,
                                        ncpu_fw_path="")
        print(' - Success')
    except kp.ApiKPException as exception:
        print('Error: upload firmware failed, error = \'{}\''.format(str(exception)))
        exit(0)

    # upload model to device
    try:
        print('[Uploading Model]')
        model_nef_descriptor = kp.core.load_model_from_file(device_group=device_group,
                                                            file_path=MODEL_FILE_PATH)
        print(' - Success')
    except kp.ApiKPException as exception:
        print('Error: upload model failed, error = \'{}\''.format(str(exception)))
        exit(0)

    #####

    evaluate_model()

    # sum total TP, FP, FN & evaluate overall Precision/Recall/F1-Score values
    print("final")
    tp_total = sum([x[0] for x in all_preds])
    fp_total = sum([x[1] for x in all_preds])
    fn_total = sum([y[0] for y in all_targets]) - tp_total

    Precision = tp_total / (tp_total + fp_total + 1e-8)
    Recall = tp_total / (tp_total + fn_total + 1e-8)
    F1_Score = 2 * Precision * Recall / (Precision + Recall + 1e-8)

    print("computing")
    # eval final mAP
    metric_result = metric.compute()

    # print/record final test results
    header = "{📊 Obj-Recognition Model Evaluation Results (YOLO format GT)}:"

    precisionStr = f"Precision: {Precision:.4f}"
    recallStr = f"Recall:    {Recall:.4f}"
    f1scoreStr = f"F1 Score:  {F1_Score:.4f}"
    mAP_resultStr = ""

    for k, v in metric_result.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                v = v.item()
            else:
                v = v.tolist()

        if isinstance(v, list):
            mAP_resultStr += f"{k}: {v}\n"
        else:
            mAP_resultStr += f"{k}: {v:.4f}\n"

    FinalResultStr = f"{header}\n\n[Test Time:{dateStr}]\n\n{mAP_resultStr}\n{precisionStr}\n{recallStr}\n{f1scoreStr}\n\n<Recognition Rate:{F1_Score * 100}%>"
    print("")
    print("=" * 50)
    print(FinalResultStr)
    print("=" * 50)

    # test 디렉토리 생성, test 결과 저장, annotation image 저장
    os.makedirs(result_dir_path, exist_ok=True)
    result_fileName = f"{result_dir_path}/test_rslt_{nowstr}.txt"
    with open(result_fileName, "w", encoding="utf-8") as f:
        f.write(FinalResultStr)
        f.close()

    #####


# 5인중 5인 탐지 성공률 집계 기능 추후 추가