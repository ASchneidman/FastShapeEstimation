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

def segment(images, model=None, use_cuda = False, background_color = 1):
    """
    image: image location
    """

    if model is None:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    to_tensor = torchvision.transforms.ToTensor()
    inp = []
    for image in images:
        tensor = to_tensor(Image.open(image))
        assert(tensor.shape[0] == 3)
        if torch.cuda.is_available() and use_cuda:
            tensor = tensor.cuda()
        inp.append(tensor)

    model.eval()
    if torch.cuda.is_available() and use_cuda:
       model = model.cuda()

    output = model(inp)
    result = []
    for i, y in enumerate(output):
        #print(f"Detected {[COCO_INSTANCE_CATEGORY_NAMES[x] for x in y['labels']]}")
        #print(f"Scores {y['scores']}")


        mask = y['masks'][0] >= 0.5
        mask = torch.cat(3 * [mask])

        inp[i][~mask] = background_color
        result.append(inp[i].detach())

    #plt.imshow(result[0].detach().cpu().permute(1,2,0).numpy())
    #plt.show()

    return result

if __name__ == '__main__':
    image = 'images/chair.jpg'
    result = segment([image])

    plt.imshow(result[0].detach().cpu().permute(1,2,0).numpy())
    plt.show()




    






