import torch
import numpy as np
from PIL import Image
from .model import get_clip_feature
from pycocotools import mask as maskUtils
from dataset.jsondataset import batch_to_device
from dataset.decode_item import sample_random_points_from_mask, sample_sparse_points_from_mask

def create_zero_input_tensors(max_objs, n_polygon_points, n_scribble_points):
    masks = torch.zeros(max_objs) # binay, indicates the instance conditioning exists or not
    text_masks = torch.zeros(max_objs) # binay, indicates the instance conditioning exists or not
    text_embeddings = torch.zeros(max_objs, 768)
    boxes_embeddings = torch.zeros(max_objs, 4)
    polygons_embeddings = torch.zeros(max_objs, n_polygon_points*2 )
    scribbles_embeddings = torch.zeros(max_objs, n_scribble_points*2 )
    segs_embeddings = torch.zeros(max_objs, 512, 512)
    points_embeddings = torch.zeros(max_objs, 2)

    return boxes_embeddings, masks, text_masks, text_embeddings, polygons_embeddings, scribbles_embeddings, segs_embeddings, points_embeddings

