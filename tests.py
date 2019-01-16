# -----------------------
#
# Copyright Jin Fagang @2018
# 
# 1/16/19
# tests
# -----------------------


from datasets.det_dataset_coco import COCODetection
from datasets.det_data_aug import PreProcessor, detection_collate
from torch.utils.data import DataLoader
import cv2
import numpy as np

root_dir = '/media/jintain/sg/permanent/datasets/coco'

train_sets = [('2017', 'train'), ('2014', 'val')]
img_dim = 512
rgb_means = (104, 117, 123)
p = 0.6


def test_data():

    ds = COCODetection(root_dir, train_sets, preproc=PreProcessor(img_dim, rgb_means, p))
    # ds = COCODetection(root_dir, train_sets)
    train_loader = DataLoader(ds, batch_size=3, shuffle=True, collate_fn=detection_collate)
    # train_loader = DataLoader(ds, batch_size=3, shuffle=True)

    for item in train_loader:
        img_tensor, label_tensor = item

        print(label_tensor[0].cpu().numpy())
        print(img_tensor[0].cpu().numpy().shape)
        cv2.imshow('rr', np.transpose(img_tensor[0].cpu().numpy(), [1, 2, 0]))
        cv2.waitKey(0)


if __name__ == '__main__':
    test_data()




