import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Identity(nn.Module):
    # Esta clase la usaremos para reemplazar las capas que deseemos por la identidad
    # Esto no hace nada y es como eliminar la capa deseada
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def topk_classes(input_probs, k):
    """
    Devuelve las k clases mas votadas de la entrada de mayor a menor probabilidad
    Para uso mas exhaustivo ir a https://pytorch.org/docs/stable/torch.html#torch.topk
    Ejemplo: si pasamos algo como [2, 4, 1, 8] con k=3 devolveria [3, 1, 0]
    """
    if type(input_probs) == np.ndarray:
        input_probs = torch.from_numpy(input_probs)
    probs_values, class_indxs = torch.topk(input_probs, k)
    return class_indxs


''' ######################################################################## '''
''' ####################### GAUSSIAN NOISE LAYER ########################### '''
''' ######################################################################## '''


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, device=None, is_relative_detach=True):
        super().__init__()
        if device is None: device = DEVICE
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device).float()

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


''' ######################################################################## '''
''' ########################### ENSEMBLING ################################# '''
''' ######################################################################## '''


def models_average(outputs, scheme, vector_solution=True):
    """
    Dada una lista de salidas (outputs) a las que se 
    les aplica previamente la funcion softmax promedia dichas salidas
    teniendo dos posibilidades:
        - voting: donde la clase de mayor probabilidad vota 1 y el resto 0
        - sum: se suma la probabilidad de todas las clases para decidir
    vector_solution indica si devolvemos un vector con las clases de mayor valor
    o el resultado de aplicar el esquema
    """
    if not (type(outputs) is list or type(outputs) is tuple):
        assert False, "List of diferents outputs needed!"

    # El esquema de la suma sumamos las probabilidades
    # de las salidas que se nos proporcionan
    if scheme == "sum":
        result = outputs[0].clone()
        for output in outputs[1:]:
            result += output.clone()

    # En el esquema por votacion cada clase vota su clase de mayor probabilidad
    # y finalmente la clase mas votada por las salidas es la resultante
    elif scheme == "voting":
        # one_zeros es la matriz de salida transformada a 1 para la clase de mayor
        # probabilidad y 0s en el resto
        one_zeros = (outputs[0].clone() == outputs[0].clone().max(dim=1)[:, None]).astype(int)
        result = one_zeros
        for output in outputs[1:]:
            one_zeros = (output == output.max(dim=1)[:, None]).astype(int)
            result += one_zeros.clone()

    else:
        assert False, "Ivalid model average scheme!"

    if vector_solution: return np.argmax(result, axis=1)
    return result
