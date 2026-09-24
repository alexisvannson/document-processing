# document-processing


You have three fundamentally different paths for receipt text detection. Moving away from Faster R-CNN changes how you frame the computer vision problem: from **region proposals** (Faster R-CNN), to **grid-based regression** (YOLO), to **pixel segmentation** (UNet), or to **specialized text algorithms**.

Here is how UNet, YOLO, and the industry-standard OCR models compare for your specific use case.

## 1. The YOLO Route (One-Stage Object Detection)

Instead of a two-step process (find regions -> classify them), YOLO looks at the entire image at once and predicts bounding boxes over a grid.

* **How you use it:** You treat lines of text (or individual words) exactly like objects (like cars or dogs). The Ultralytics YOLOv8 or YOLOv11 frameworks are the standard here.
* **The Receipt Advantage:** YOLO is incredibly fast and highly optimized for deployment. If your receipt photos are generally straight, standard YOLO is excellent. If they are wrinkled or captured at angles, you can use **YOLO-OBB (Oriented Bounding Boxes)**, which predicts rotated rectangles.
* **The Catch:** Standard YOLO anchors can sometimes struggle with extreme aspect ratios (e.g., a very long, thin line of text across an entire receipt).

## 2. The UNet Route (Semantic Segmentation)

Using a UNet Abandons bounding boxes during the neural network phase entirely. Instead, the model acts like a highlighter, predicting a binary mask where every pixel is classified as "Text" (1) or "Background" (0).

* **How you use it:** You must convert your bounding box data into solid white polygons on a black background to train the UNet.
* **The Receipt Advantage:** UNet does not care about aspect ratios, rotations, or curved lines. If a receipt is crumpled and the text bends, UNet simply highlights the curved text perfectly.
* **The Catch:** The neural network outputs a pixel mask, not coordinates. You must write heavy post-processing scripts (using OpenCV's `cv2.findContours` or `cv2.minAreaRect`) to convert the predicted white pixel blobs back into mathematical bounding boxes for your text reader.

## 3. The Specialized OCR Route (The Industry Standard)

While YOLO and UNet are general-purpose computer vision tools, the OCR industry uses specialized architectures built specifically for reading documents and receipts.

* **DBNet (Differentiable Binarization):** This is a segmentation-based model (similar in spirit to UNet) but heavily optimized for text boundaries. It is exceptionally good at separating lines of text that are very close together, which is a common failure point for standard UNet.
* **CRAFT (Character Region Awareness for Text):** Instead of detecting words or lines, CRAFT detects individual *characters* and the spatial links between them. It is highly robust to weird receipt fonts, extreme spacing, and curved text.

---

### Strategy Comparison

| Architecture | Paradigm | Best Feature for Receipts | Major Drawback |
| --- | --- | --- | --- |
| **Faster R-CNN (ResNet+FPN)** | Two-Stage Detection | High accuracy, great at handling varying text scales. | Slower inference time, heavy memory footprint. |
| **YOLO (v8 / v11)** | One-Stage Detection | Unmatched speed, massive ecosystem, easy deployment. | Struggles with extreme aspect ratios and dense packing. |
| **UNet** | Pixel Segmentation | Conforms perfectly to curved, crumpled, or angled text. | Requires heavy post-processing to get bounding boxes. |
| **DBNet / CRAFT** | Specialized Text AI | State-of-the-art for documents; handles close/dense text perfectly. | Steeper learning curve; requires specialized frameworks (e.g., MMOCR, PaddleOCR). |



# docTR (db_resnet50) and PaddleOCR both ship pretrained DBNet models you can fine-tune on these 200 receipts.  