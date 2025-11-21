import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("/runs/obb/train/weights/best.pt")

# Load image path
image_path = ".jpg"

# Predict Display Values -> ปิดตรงนี้ถ้าต้องการเซฟดูรูปภาพ
results = model(image_path)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

# for result in results:
#     if result.obb is not None:
#         xywhr = result.obb.xywhr  # center-x, center-y, width, height, rotation
#         xyxyxyxy = result.obb.xyxyxyxy  # 4-point polygon
#         names = [result.names[cls.item()] for cls in result.obb.cls.int()]
#         confs = result.obb.conf
#         # แสดงผล
#         for i in range(len(xyxyxyxy)):
#             print(f'Class name: {names[i]}, Conf: {confs[i].item():.2f}, Polygon: {xyxyxyxy[i].tolist()}, xywhr: {xywhr[i].tolist()}')
#     else:
#         print("No OBB result found for this image.")

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

img = cv2.imread(image_path)

# Draw results
for result in results:
    if result.obb is not None:
        xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()  # shape: [N, 8]
        names = [result.names[cls.item()] for cls in result.obb.cls.int()]
        confs = result.obb.conf.cpu().numpy()

        for i in range(len(xyxyxyxy)):
            poly = xyxyxyxy[i].reshape((4, 2)).astype(np.int32)
            cv2.polylines(img, [poly], isClosed=True, color=(0, 255, 0), thickness=1)
            label = f"{names[i]} {confs[i]:.2f}"
            cv2.putText(img, label, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        print("No OBB result found for this image.")

# Save the image
output_path = "./inference_output.jpg"
cv2.imwrite(output_path, img)
print(f"Saved image with OBBs to {output_path}")
