import torch, torchvision
from PIL import Image
import matplotlib.pyplot as plt

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def segment(image, model=None):
    """
    image: image location
    """

    if model is None:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    to_tensor = torchvision.transforms.ToTensor()
    inp = []
    tensor = to_tensor(Image.open(image))
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    inp.append(tensor)

    model.eval()
    if torch.cuda.is_available():
       model = model.cuda()

    output = model(inp)[0]
    print(output['labels'])
    print(f"Detected {COCO_INSTANCE_CATEGORY_NAMES[output['labels'][0]]}")

    mask = output['masks'][0] >= 0.5
    mask = torch.cat(3 * [mask])

    inp[0][~mask] = 1
    return inp[0]

    return mask * inp[0]

if __name__ == '__main__':
    image = 'images/chair.jpg'
    result = segment(image)

    plt.imshow(result.detach().cpu().permute(1,2,0).numpy())
    plt.show()




    






