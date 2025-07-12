#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import random
import PIL
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from PIL  import ImageDraw
import os
import os.path as op
import gc
import json
from typing import List
import logging

try:
    from .blob_storage import BlobStorage, disk_usage
except:
    class BlobStorage:
        pass
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo)


def imagenet_deprocess(rescale_image=True):
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
        T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)


def imagenet_deprocess_batch(imgs, rescale=True):
    """
    Input:
    - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images

    Output:
    - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
      in the range [0, 255]
    """
    if isinstance(imgs, torch.autograd.Variable):
        imgs = imgs.data
    imgs = imgs.cpu().clone()
    deprocess_fn = imagenet_deprocess(rescale_image=rescale)
    imgs_de = []
    for i in range(imgs.size(0)):
        img_de = deprocess_fn(imgs[i])[None]
        img_de = img_de.mul(255).clamp(0, 255).byte()
        imgs_de.append(img_de)
    imgs_de = torch.cat(imgs_de, dim=0)
    return imgs_de


class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)


def unpack_var(v):
    if isinstance(v, torch.autograd.Variable):
        return v.data
    return v


def split_graph_batch(triples, obj_data, obj_to_img, triple_to_img):
    triples = unpack_var(triples)
    obj_data = [unpack_var(o) for o in obj_data]
    obj_to_img = unpack_var(obj_to_img)
    triple_to_img = unpack_var(triple_to_img)

    triples_out = []
    obj_data_out = [[] for _ in obj_data]
    obj_offset = 0
    N = obj_to_img.max() + 1
    for i in range(N):
        o_idxs = (obj_to_img == i).nonzero().view(-1)
        t_idxs = (triple_to_img == i).nonzero().view(-1)

        cur_triples = triples[t_idxs].clone()
        cur_triples[:, 0] -= obj_offset
        cur_triples[:, 2] -= obj_offset
        triples_out.append(cur_triples)

        for j, o_data in enumerate(obj_data):
            cur_o_data = None
            if o_data is not None:
                cur_o_data = o_data[o_idxs]
            obj_data_out[j].append(cur_o_data)

        obj_offset += o_idxs.size(0)

    return triples_out, obj_data_out


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    WW, HH = pil_image.size

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)

    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    # at this point, the min of pil_image side is desired image_size
    performed_scale = image_size / min(WW, HH)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2

    info = {"performed_scale": performed_scale, 'crop_y': crop_y, 'crop_x': crop_x, "WW": WW, 'HH': HH}

    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size], info


def random_crop_arr(pil_image, image_size, min_crop_frac=1.0, max_crop_frac=1.0):
    WW, HH = pil_image.size
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    # 裁剪后的整体尺寸和短边相同，所以长边会溢出，之后要裁剪掉
    # at this point, the min of pil_image side is desired image_size
    performed_scale = image_size / min(WW, HH)
    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    info = {"performed_scale": performed_scale, 'crop_y': crop_y, 'crop_x': crop_x, "WW": WW, 'HH': HH}
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size], info


def recalculate_box_and_verify_if_valid(x, y, w, h, trans_info, image_size, min_box_size):
    """
    x,y,w,h:  the original annotation corresponding to the raw image size.
    trans_info: what resizing and cropping have been applied to the raw image
    image_size:  what is the final image size
    """

    x0 = x * trans_info["performed_scale"] - trans_info['crop_x']
    y0 = y * trans_info["performed_scale"] - trans_info['crop_y']
    x1 = (x + w) * trans_info["performed_scale"] - trans_info['crop_x']
    y1 = (y + h) * trans_info["performed_scale"] - trans_info['crop_y']

    # at this point, box annotation has been recalculated based on scaling and cropping
    # but some point may fall off the image_size region (e.g., negative value), thus we
    # need to clamp them into 0-image_size. But if all points falling outsize of image
    # region, then we will consider this is an invalid box.
    valid, (x0, y0, x1, y1) = to_valid(x0, y0, x1, y1, image_size, min_box_size)

    if valid:
        # we also perform random flip.
        # Here boxes are valid, and are based on image_size
        if trans_info["performed_flip"]:
            x0, x1 = image_size - x1, image_size - x0

    return valid, (x0, y0, x1, y1)


def to_valid(x0, y0, x1, y1, image_size, min_box_size):
    valid = True

    if x0 > image_size or y0 > image_size or x1 < 0 or y1 < 0:
        valid = False  # no way to make this box vide, it is completely cropped out
        return valid, (None, None, None, None)

    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, image_size)
    y1 = min(y1, image_size)

    if (x1 - x0) * (y1 - y0) / (image_size * image_size) < min_box_size:
        valid = False
        return valid, (None, None, None, None)

    return valid, (x0, y0, x1, y1)


def draw_box(img, boxes):
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    for bid, box in enumerate(boxes):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=colors[bid % len(colors)], width=4)
        # draw.rectangle([box[0], box[1], box[2], box[3]], outline ="red", width=2) # x0 y0 x1 y1
    return img


def generate_lineidx(filein: str, idxout: str) -> None:
    idxout_tmp = idxout + '.tmp'
    with open(filein, 'r') as tsvin, open(idxout_tmp, 'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos != fsize:
            tsvout.write(str(fpos) + "\n")
            tsvin.readline()
            fpos = tsvin.tell()
    os.rename(idxout_tmp, idxout)


def read_to_character(fp, c):
    result = []
    while True:
        s = fp.read(32)
        assert s != ''
        if c in s:
            result.append(s[: s.index(c)])
            break
        else:
            result.append(s)
    return ''.join(result)


class TSVFile(object):
    def __init__(self,
                 tsv_file: str,
                 if_generate_lineidx: bool = False,
                 lineidx: str = None,
                 class_selector: List[str] = None,
                 blob_storage: BlobStorage = None):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx' \
            if not lineidx else lineidx
        self.linelist = op.splitext(tsv_file)[0] + '.linelist'
        self.chunks = op.splitext(tsv_file)[0] + '.chunks'
        self._fp = None
        self._lineidx = None
        self._sample_indices = None
        self._class_boundaries = None
        self._class_selector = class_selector
        self._blob_storage = blob_storage
        self._len = None
        # the process always keeps the process which opens the file.
        # If the pid is not equal to the currrent pid, we will re-open the file.
        self.pid = None
        # generate lineidx if not exist
        if not op.isfile(self.lineidx) and if_generate_lineidx:
            generate_lineidx(self.tsv_file, self.lineidx)

    def __del__(self):
        self.gcidx()
        if self._fp:
            self._fp.close()
            # physically remove the tsv file if it is retrieved by BlobStorage
            if self._blob_storage and 'azcopy' in self.tsv_file and os.path.exists(self.tsv_file):
                try:
                    original_usage = disk_usage('/')
                    os.remove(self.tsv_file)
                    logging.info("Purged %s (disk usage: %.2f%% => %.2f%%)" %
                                 (self.tsv_file, original_usage, disk_usage('/') * 100))
                except:
                    # Known issue: multiple threads attempting to delete the file will raise a FileNotFound error.
                    # TODO: try Threadling.Lock to better handle the race condition
                    pass

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def gcidx(self):
        logging.debug('Run gc collect')
        self._lineidx = None
        self._sample_indices = None
        # self._class_boundaries = None
        return gc.collect()

    def get_class_boundaries(self):
        return self._class_boundaries

    def num_rows(self, gcf=False):
        if (self._len is None):
            self._ensure_lineidx_loaded()
            retval = len(self._sample_indices)

            if (gcf):
                self.gcidx()

            self._len = retval

        return self._len

    def seek(self, idx: int):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[self._sample_indices[idx]]
        except:
            logging.info('=> {}-{}'.format(self.tsv_file, idx))
            raise
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_first_column(self, idx: int):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return read_to_character(self._fp, '\t')

    def get_key(self, idx: int):
        return self.seek_first_column(idx)

    def __getitem__(self, index: int):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            logging.debug('=> loading lineidx: {}'.format(self.lineidx))
            with open(self.lineidx, 'r') as fp:
                lines = fp.readlines()
                lines = [line.strip() for line in lines]
                self._lineidx = [int(line) for line in lines]

            # read the line list if exists
            linelist = None
            if op.isfile(self.linelist):
                with open(self.linelist, 'r') as fp:
                    linelist = sorted(
                        [
                            int(line.strip())
                            for line in fp.readlines()
                        ]
                    )

            if op.isfile(self.chunks):
                self._sample_indices = []
                self._class_boundaries = []
                class_boundaries = json.load(open(self.chunks, 'r'))
                for class_name, boundary in class_boundaries.items():
                    start = len(self._sample_indices)
                    if class_name in self._class_selector:
                        for idx in range(boundary[0], boundary[1] + 1):
                            # NOTE: potentially slow when linelist is long, try to speed it up
                            if linelist and idx not in linelist:
                                continue
                            self._sample_indices.append(idx)
                    end = len(self._sample_indices)
                    self._class_boundaries.append((start, end))
            else:
                if linelist:
                    self._sample_indices = linelist
                else:
                    self._sample_indices = list(range(len(self._lineidx)))

    def _ensure_tsv_opened(self):
        if self._fp is None:
            if self._blob_storage:
                self._fp = self._blob_storage.open(self.tsv_file)
            else:
                self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            logging.debug('=> re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()


class TSVWriter(object):
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        self.lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
        self.tsv_file_tmp = self.tsv_file + '.tmp'
        self.lineidx_file_tmp = self.lineidx_file + '.tmp'

        self.tsv_fp = open(self.tsv_file_tmp, 'w')
        self.lineidx_fp = open(self.lineidx_file_tmp, 'w')

        self.idx = 0

    def write(self, values, sep='\t'):
        v = '{0}\n'.format(sep.join(map(str, values)))
        self.tsv_fp.write(v)
        self.lineidx_fp.write(str(self.idx) + '\n')
        self.idx = self.idx + len(v) + 1

    def close(self):
        self.tsv_fp.close()
        self.lineidx_fp.close()
        os.rename(self.tsv_file_tmp, self.tsv_file)
        os.rename(self.lineidx_file_tmp, self.lineidx_file)
