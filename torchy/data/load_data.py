import os

import numpy as np
import torch
from skimage import io
from torch.utils import data


def load_img(path):
    # return np.array(PIL.Image.open(path).convert('RGB'))
    return io.imread(path)


class FoldersDataset(data.Dataset):
    """
        Cargador de prueba de dataset almacenado en carpetas del modo
            -> train/clase1/TodasImagenes ...
        data_path: ruta a la carpeta padre del dataset. Ejemplo train/
        transforms: lista de transformaciones de albumentations a aplicar
        cat2class: diccionario con claves clas clases y valor la codificacion
            de cada clase. Ejemplo {'perro':0, 'gato':1}
    """

    def __init__(self, data_path, transforms=[], cat2class=[], normalization=""):
        different_classes, all_paths, all_classes = [], [], []
        for path, subdirs, files in os.walk(data_path):
            for name in files:
                fullpath = os.path.join(path, name)
                current_class = fullpath.split("/")[-2]
                all_paths.append(fullpath)
                all_classes.append(current_class)
                if current_class not in different_classes: different_classes.append(current_class)

        if cat2class == []: cat2class = dict(zip(different_classes, np.arange(0, len(different_classes))))

        for indx, c_class in enumerate(all_classes):
            all_classes[indx] = cat2class[c_class]

        self.imgs_paths = all_paths
        self.labels = all_classes
        self.transforms = transforms
        self.norm = normalization

    def __getitem__(self, index):
        img = load_img(self.imgs_paths[index])

        if self.transforms != []:
            for transform in self.transforms:
                img = apply_img_albumentation(transform, img)

        img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1))
        if self.norm != "": img = normalize_data(img, self.norm)
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs_paths)


def normalize_data(feats, norm):
    if norm == '0_1range':
        max_val = feats.max()
        min_val = feats.min()
        feats -= min_val
        feats /= (max_val - min_val)


    elif norm == '-1_1range' or norm == 'np_range':
        max_val = feats.max()
        min_val = feats.min()
        feats *= 2
        feats -= (max_val - min_val)
        feats /= (max_val - min_val)

    elif norm == '255':
        feats /= 255

    elif norm == None or norm == "":
        pass
    else:
        assert False, "Data normalization not implemented: {}".format(norm)

    return feats


def apply_img_albumentation(aug, image):
    image = aug(image=image)['image']
    return image
