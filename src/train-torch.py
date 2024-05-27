import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class Fathom24Dataset(Dataset):
    """Fathom2024 dataset."""

    def __init__(self, json_file, root_dir, transform=None):
        """
        Arguments:
            json_file (string): Path to the json file with COCO annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(json_file, "rb") as f:
            annotations = json.load(f)
        self.images_df = pd.DataFrame(annotations["images"])
        self.annotations_df = pd.DataFrame(annotations["annotations"]).set_index("id")
        self.categories_df = pd.DataFrame(annotations["categories"]).set_index("id")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.images_df.loc[idx].id
        img_name = os.path.join(self.root_dir,
                                self.images_df.loc[idx, "file_name"])
        image = io.imread(img_name)
        annotations = self.annotations_df[self.annotations_df.image_id == img_id]
        bboxes = annotations.bbox.explode().to_numpy(dtype=float).reshape(-1, 4)
        # Faster RCNN expects boxes as xmin, ymin, xmax, ymax
        bboxes_transformed = bboxes[:,:2]
        bboxes_transformed = np.hstack((bboxes_transformed, (bboxes[:,0] + bboxes[:,2]).reshape(-1, 1)))
        bboxes_transformed = np.hstack((bboxes_transformed, (bboxes[:,1] + bboxes[:,3]).reshape(-1, 1)))
        cat_ids = annotations.category_id.to_numpy(dtype=int)

        annotations = {'boxes': bboxes_transformed, 'labels': cat_ids}

        sample = (image, annotations)

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bboxes, cats = sample[0], sample[1]['boxes'], sample[1]['labels']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        bboxes = bboxes * [new_w / w, new_h / h, new_w / w, new_h / h]

        return img, {'boxes': bboxes, 'labels': cats}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bboxes, cats = sample[0], sample[1]['boxes'], sample[1]['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float(), {
            'boxes': torch.from_numpy(bboxes), 
            'labels': torch.from_numpy(cats)
            }

def collate_fn(batch):  # needed for dictionary data
    return tuple(zip(*batch))

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

if __name__ == "__main__":
    transformed_f24_dataset = Fathom24Dataset(json_file="../data/train.json",
                                              root_dir="../data/train",
                                              transform=transforms.Compose([
                                                Rescale((256, 256)),
                                                ToTensor()
                                              ]))
    data_loader = DataLoader(transformed_f24_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 19
    num_epochs = 2
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    len_dataloader = len(data_loader)
    for epoch in range(num_epochs):
        model.train()
        i = 0    
        for imgs, annotations in data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')