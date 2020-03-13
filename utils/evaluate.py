import json
import cv2
import numpy as np
import sklearn.metrics
import pickle
import os
import matplotlib.pyplot as plt

def visualize(cam, boxes, vid, frame, cls, keys):
    img = os.path.join('/media/yuxi/Data/AudioVisual/data/Audioset/val', keys[vid],
                       str(frame).zfill(4)+'.jpg')
    img = cv2.imread(img)
    img = cv2.resize(img, (256, 256))
    cam = cam / np.max(cam+np.spacing(1))
    cam = cam * 255
    cam = cv2.applyColorMap(cam.astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    cam = cam / 255
    for box in boxes:
        img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                            (int(box[0]+box[2]), int(box[1]+box[3])), (0, 255, 0), 2)
    img = img[:, :, ::-1] / 255
    plt.imsave('vis/'+str(vid)+'_'+str(frame)+'_'+str(cls)+'.jpg', 0.5*cam+0.5*img)


effidx = np.load('audioset_val.npy')
gt = json.load(open('normav.json', 'r'))

infermap = np.load('agnostic.npy')
vids = pickle.load(open('/media/yuxi/Data/AudioVisual/data/Audioset/val_audio.pkl', 'rb'),
                   encoding='bytes')
keys = list(vids)
keys.sort()

def eval_agnostic():
    annotations = dict()
    for item in gt['annotations']:
        id = item['image_id']
        cls = item['category_id']
        if id not in annotations:
            annotations[id] = dict()
        if cls not in annotations[id]:
            annotations[id][cls] = []
        annotations[id][cls].append(item['bbox'])

    cious = []
    for k in annotations:
        ciou = 0.0
        vid = k // 10
        frame = k % 10
        label = annotations[k]
        if len(label) != 2:
            continue
        gtmap = np.zeros((256, 256))
        boxes = []
        for cls in label:
            boxes += label[cls]
        boxes = np.array(boxes) * 256
        for box in boxes:
            gtmap[int(box[1]): int(box[1] + box[3]), int(box[0]): int(box[0] + box[2])] = 1.0
        infer = infermap[vid, frame]
        infer = cv2.resize(infer, (256, 256))
        infer[infer < 0] = 0
        visualize(infer, boxes, vid, frame, -1, keys)
        ciou += np.sum((gtmap > 0) * (infer > 0.1 * np.max(infer))) / \
                (np.sum(gtmap) + np.sum((gtmap == 0) * (infer > 0.1 * np.max(infer))))

        cious.append(ciou)
    cious = np.array(cious)
    ciou = np.sum(cious >= 0.5) / len(cious)
    results = []
    for i in range(21):
        result = np.sum(np.array(cious) >= 0.05 * i)
        result = result / len(cious)
        results.append(result)
    x = [0.05 * i for i in range(21)]
    auc = sklearn.metrics.auc(x, results)
    print(ciou, auc)

def class_eval():
    annotations = dict()
    for item in gt['annotations']:
        id = item['image_id']
        cls = item['category_id']
        if id not in annotations:
            annotations[id] = dict()
        if cls not in annotations[id]:
            annotations[id][cls] = []
        annotations[id][cls].append(item['bbox'])

    cious = []
    for k in annotations:
        ciou = 0.0
        vid = k // 10
        frame = k % 10
        label = annotations[k]
        if len(label) != 2:
            continue
        for cls in label:
            boxes = label[cls]
            boxes = np.array(boxes) * 256
            gtmap = np.zeros((256, 256))
            for box in boxes:
                gtmap[int(box[1]): int(box[1]+box[3]), int(box[0]): int(box[0]+box[2])] = 1.0
            infer = infermap[vid, frame, cls]
            infer = cv2.resize(infer, (256, 256))
            infer[infer<0] = 0
            visualize(infer, boxes, vid, frame, cls, keys)
            ciou += np.sum((gtmap>0)*(infer>0.1*np.max(infer))) / \
                   (np.sum(gtmap)+np.sum((gtmap==0)*(infer>0.1*np.max(infer))))
        ciou /= len(label)
        cious.append(ciou)
    cious = np.array(cious)
    ciou = np.sum(cious>=0.5) / len(cious)
    results = []
    for i in range(21):
        result = np.sum(np.array(cious) >= 0.05 * i)
        result = result / len(cious)
        results.append(result)
    x = [0.05 * i for i in range(21)]
    auc = sklearn.metrics.auc(x, results)
    print(ciou, auc)

#eval_agnostic()
#class_eval()
