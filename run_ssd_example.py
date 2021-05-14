from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import torch
import os

def _find_classes(dir):

    if sys.version_info >= (3, 5):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def sortBoxes(Boxes,labels):
    line1=[]
    line2=[]
    labelsline1=[]
    labelsline2=[]
    BoxesSort=[]
    LabelsSort=[]
    maxBoxes=torch.max(Boxes,0)
    minBoxes=torch.min(Boxes,0)
    if maxBoxes.values[1] > (Boxes[0][3].item()-Boxes[0][1].item()):
        thresY=(maxBoxes.values[1]+minBoxes.values[1])/2
        for i in range(boxes.size(0)):
            box = boxes[i, :].numpy()
            if box[1]<thresY.item():
                line1.append(box)
                labelsline1.append(labels.numpy()[i])
            else:
                line2.append(box)
                labelsline2.append(labels.numpy()[i])

        sortline1=sorted(line1 , key=lambda k:k[0])
        sortline2=sorted(line2 , key=lambda k:k[0])
        indexaftersortl1=[i[0] for i in sorted(enumerate(line1), key=lambda x:x[1][0])]
        indexaftersortl2=[i[0] for i in sorted(enumerate(line2), key=lambda x:x[1][0])]
        BoxesSort=sortline1+sortline2
        for i in indexaftersortl1:
            LabelsSort.append(labelsline1[i])
        for i in indexaftersortl2:
            LabelsSort.append(labelsline2[i])
    else:
        BoxesSort=sorted(Boxes.numpy() , key=lambda k:k[0])
        indexaftersortl=[i[0] for i in sorted(enumerate(Boxes.numpy()), key=lambda x:x[1][0])]
        for i in indexaftersortl:
            LabelsSort.append(labels.numpy()[i])
    print("LabelsSort",LabelsSort)
    return torch.tensor(BoxesSort),torch.tensor(LabelsSort)

if len(sys.argv) < 5:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
image_path = sys.argv[4]

class_names = [name.strip() for name in open(label_path).readlines()]

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

classes, class_to_idx = _find_classes(os.path.join("images","images_long"))
for target in sorted(class_to_idx.keys()):
    d = os.path.join(os.path.join("images","images_long"),target)
    if not os.path.isdir(d):
        continue
    for folder in sorted(os.listdir(d)):
        orig_image = cv2.imread(d+"/"+folder)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = predictor.predict(image, 10, 0.3)
        boxes, labels=sortBoxes(boxes,labels)
        labels_dict=[]
        #print(boxes)
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            #print(box)
            cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 1)
            #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
            label = f"{class_names[labels[i]]}"
            labels_dict.append(label)
            print(label)
            cv2.putText(orig_image, label,
                (int(box[0]) , int(box[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # font scale
                (0, 0, 255),
                1)  # line type
        print(labels_dict)
        print(f"Found {len(probs)} objects")
        cv2.imshow("image",orig_image)
        cv2.waitKey(0)
 
