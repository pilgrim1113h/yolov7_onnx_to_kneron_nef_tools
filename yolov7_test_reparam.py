import numpy as np
import onnxruntime
from PIL import Image, ImageOps, ImageDraw, ImageFont

ANCHORS = [[(12, 16), (19, 36), (40, 28)],
           [(36, 75), (76, 55), (72, 146)],
           [(142, 110), (192, 243), (459, 401)]]

STRIDES = [8, 16, 32]

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']

rng = np.random.default_rng(3)
colors = (rng.uniform(0, 255, size=(len(class_names), 3))).astype(np.uint8)  # different colors for different classes


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

    # conver to np array & normalize
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


#
if __name__ == "__main__":
    #
    img_path = r"000000128372.jpg"

    model_path = r"reparam_kyv7_00000000.onnx"  # reparameterized model only
   # model_path = r"v7test.onnx"  # reparameterized model only
    conf_thres = 0.25
    iou_thres = 0.45

    # init onnx model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] # if use_cuda else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(model_path, providers=providers)
    inp = session.get_inputs()[0]
    input_name = inp.name    # input_shape = (1, 3, 640, 640)

    # preprocess
    letterbox_img, pad_lt, scale = preprocess(img_path)

    # run inference
    outputs = session.run(None, {input_name: letterbox_img})

    # output decoding
    print(len(outputs))      # 3  -> stride 8/16/32에 대응하는 3개 scale map
    print(outputs[0].shape)  # (1, 33, 80, 80)  (stride  8   8 * 80 = 640)
    print(outputs[1].shape)  # (1, 33, 40, 40)  (       16  16 * 40 = 640)
    print(outputs[2].shape)  # (1, 33, 20, 20)  (       32  32 * 20 = 640)
    # YOLOv7의 3개 detection head(P3, P4, P5) 에서 나온 출력이다. 이 3개의 서로 다른 해상도 feature map을 이용해 작은 물체~큰 물체까지 모두 탐지한다.
    # 본래 PyTorch에서는 출력 shape이 다음이어야 한다:
    #
    # (1, 3, 80, 80, 11)  (p3 작은 객체
    # (1, 3, 40, 40, 11)  (p4 중간 객체
    # (1, 3, 20, 20, 11)  (p5  큰 객체       <- 11은 x, y, w, h, class_id, + 6 class_conf  총 11개의 정보
    #
    # 그러나 reparameterization을 수행한 pt모델을 ONNX로 export 하는 과정에서 anchor 차원(3) × channel(11)이 합쳐져서 위와 같이 4채널로 나오기에
    # to_5d 함수로 채널 / 출력을 나눠서 차원을 다시 구성함

    # YOLOv7의 Detect 레이어가 ONNX export 시 다음 연산을 적용했기 때문:
    #
    # (na, no, H, W) → (na * no, H, W)
    #
    # 이는 reshape + flatten + transpose가 fused된 결과
    #
    # 즉 ONNX simplify를 거치면 anchor dimension을 살리지 않고 단일 channel axis로 합침.
    boxes, scores, class_ids = decode_onnx_outputs(outputs, conf_thres)

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
        final_img = draw_detections(img_path, boxes, scores, class_ids, pad_lt, scale)
        final_img.show()

    #


