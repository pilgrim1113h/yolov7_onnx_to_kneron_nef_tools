import onnx
import ktc
import kneronnxopt
import os
import sys
import numpy as np

sys.path.append("/docker_mount/")
from yolov7_test_reparam import *  # v7
# from pre_post_process import *   # v11
from PIL import Image, ImageDraw

#
original_m = onnx.load("/docker_mount/reparam_kyv7_00000000.onnx")
opt_m = kneronnxopt.optimize(original_m)
onnx.save(opt_m, "/data1/optimized_kyv7_00000000.onnx")

#
km = ktc.ModelConfig(32769, "0001", "730", onnx_model=opt_m)
eval_result = km.evaluate()

# prepare preprocessed datas (for test & quantization)
dataset_dir = "/docker_mount/test_dataset/quantization/"
image_paths = [dataset_dir + img for img in os.listdir(dataset_dir)]
preprocessed_data = []
for i, path in enumerate(image_paths):
    print(i)
    data = preprocess(path)
    preprocessed_data.append(data)
# 위 작업은 오래 걸릴 수 있음,  컨테이너 종료후 다시 처음 시작했을때 빠르게 된다


# 전처리 데이터 결과에서 첫째 요소들만 모아서 별도의 이미지 넘파이 어레이 생성
preprocessed_images = [data[0] for data in preprocessed_data]
input_mapping = {"images": preprocessed_images}

# optimized onnx test
class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'bus', 5: 'truck'}
image_idx = 30
img = Image.open(image_paths[image_idx])
draw = ImageDraw.Draw(img)

test_result_opt_onnx = ktc.kneron_inference(preprocessed_images[image_idx],
                                            onnx_file="/data1/optimized_kyv7_00000000.onnx", input_names=["images"])

# for v7
boxes, scores, class_ids = decode_onnx_outputs(test_result_opt_onnx, 0.25)
boxes, scores, class_ids = postprocess(boxes, scores, class_ids)
final_img = draw_detections(image_paths[image_idx], boxes, scores, class_ids, preprocessed_data[image_idx][1],
                            preprocessed_data[image_idx][2])
final_img.show()
final_img.save("/docker_mount/yv7_opt_onnx_test_output.jpg")

<<<<<<< HEAD
#
# 이건 v11용..
#
#onnx_test_output = np.array(test_result_opt_onnx[0])
#onnx_test_output = onnx_test_output[0].transpose(1, 0)
#opt_onnx_result = postprocess(onnx_test_output, class_names=class_names, pad_wh=preprocessed_data[image_idx][1],
#                              scale=preprocessed_data[image_idx][2], orig_width=img.size[0], orig_height=img.size[1])

#for (x1, y1, x2, y2, conf, cls_name) in opt_onnx_result:
#    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
#img.show()
#
#
=======
>>>>>>> 6626829e4c2d0ec86fbe0688043b6708b60ae210

# analysis 는 compile 시 quantization을 위해 필요  - 약 1시간 소요됨
bie_path = km.analysis(input_mapping, threads=8)

# bie test
fixed_result = ktc.kneron_inference(preprocessed_images[image_idx], bie_file=bie_path, input_names=["images"],
                                    platform=730)  # <-오류로 튕길 때 있음.. 그러므로 compile을 먼저 할 것

# nef compile
nef_path = ktc.compile([km])

# nef test - 다소 시간이 걸림
test_result_nef = ktc.kneron_inference(preprocessed_images[image_idx], nef_file=nef_path, input_names=["images"],
                                       platform=730)
boxes, scores, class_ids = decode_onnx_outputs(test_result_nef, 0.25)
boxes, scores, class_ids = postprocess(boxes, scores, class_ids)
final_img = draw_detections(image_paths[image_idx], boxes, scores, class_ids, preprocessed_data[image_idx][1],
                            preprocessed_data[image_idx][2])
final_img.show()
final_img.save("/docker_mount/yv7_nef_test_output.jpg")

#3:02 -    optimized_kyv7_20251206.onnx