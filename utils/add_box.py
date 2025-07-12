from PIL import ImageDraw, Image, ImageFont

colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
def yolo_to_xyxy_percentage(yolo_box):
    """
    将 YOLO 格式（百分比）的边界框转换为 xyxy 格式（百分比）
    :param yolo_box: YOLO 格式的边界框，包含 (center_x, center_y, width, height)，值范围 0 - 1
    :return: xyxy 格式的边界框，包含 (x1, y1, x2, y2)，值范围 0 - 1
    """
    # 提取 YOLO 格式的坐标
    center_x, center_y, width, height = yolo_box
    # 计算左上角和右下角的坐标
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    # 确保坐标在 0 - 1 范围内
    x1 = max(0, min(1, x1))
    y1 = max(0, min(1, y1))
    x2 = max(0, min(1, x2))
    y2 = max(0, min(1, y2))
    return x1, y1, x2, y2


def read_data(file_name):
    cats = []
    boxes = []
    with open(file_name, 'r') as f:
        for line in f:
            cat_id, x0,y0,x1,y1 = line.split(' ')
            x0,y0,x1,y1 = float(x0), float(y0), float(x1), float(y1)
            cats.append(coco_classes[int(cat_id)])
            boxes.append(yolo_to_xyxy_percentage([x0,y0,x1,y1]))

    return cats, boxes

def add_boxes(image_name=None,txt=None):
    txt_color = (255, 255, 255)
    color = (249, 65, 68)
    font = ImageFont.load_default().font_variant(size=60)
    if image_name is None:
        image = Image.new('RGB', (1024, 1024), (242,242,242))
        W, H = 1024, 1024
    else:
        image = Image.open(image_name)
        W, H = image.size
    draw = ImageDraw.Draw(image)
    # W, H = 1024, 1024

    cats, boxes = read_data(txt)
    for bid, (cat, box) in enumerate(zip(cats, boxes)):
        x0, y0, x1, y1 = box
        box = [float(x0 * W), float(y0 * H), float(x1 * W), float(y1 * H)]
        draw.rectangle(box, outline = colors[bid % len(colors)], width = 10)
        left, top, right, bottom = font.getbbox(cat)
        w = right - left +10
        h = bottom - top+10  # text width, height
        outside = box[1] - h >= 0  # label fits outside box
        draw.rectangle((box[0],
                             box[1] - h if outside else box[1],
                             box[0] + w + 1,
                             box[1] + 1 if outside else box[1] + h + 1), fill=colors[bid % len(colors)])
        # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
        draw.text((box[0], box[1] - h if outside else box[1]), cat, fill=txt_color, font=font)
    image.save('./results_429623.jpg')

if __name__ == '__main__':
    add_boxes( None, 'd:/datasets/coco/labels/val2017/000000429623.txt')