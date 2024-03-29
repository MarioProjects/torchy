import PIL
import math
import numpy as np
import torch

CROSS_ENTROPY_ONE_HOT_WARNING = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_model_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def defrost_model_params(model):
    # Funcion para descongelar redes!
    for param in model.parameters():
        param.requires_grad = True


''' ######################################################################## '''
''' ######################## GENERAL TRAINING ############################## '''
''' ######################################################################## '''


def train_step(train_loader, model, criterion, optimizer):
    train_loss, train_correct = [], 0
    model.train()
    for image, target in train_loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        image = image.type(torch.float)
        y_pred = model(image)
        loss = criterion(y_pred.float(), target.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred = y_pred.max(1)  # get the index of the max log-probability
        train_correct += pred.eq(target).sum().item()
        train_loss.append(loss.item())

    train_accuracy = 100. * train_correct / len(train_loader.dataset)
    return np.mean(train_loss), train_accuracy


def val_step(val_loader, model, criterion, data_predicts = False):
    val_loss, val_correct = [], 0
    predicts, truths = [], []
    init = -1
    model.eval()
    with torch.no_grad():
        for image, target in val_loader:
            image, target = image.to(DEVICE), target.to(DEVICE)
            image = image.type(torch.float)
            y_pred = model(image)

            loss = criterion(y_pred.float(), target.long())
            val_loss.append(loss.item())
            _, pred = y_pred.max(1)  # get the index of the max log-probability
            val_correct += pred.eq(target).sum().item()
            if data_predicts:
                predicts.append(pred.detach().cpu().numpy())
                truths.append(target.detach().cpu().numpy())

    val_accuracy = 100. * val_correct / len(val_loader.dataset)
    if not data_predicts: return np.mean(val_loss), val_accuracy

    predicts = np.concatenate(predicts)
    truths = np.concatenate(truths)
    return np.mean(val_loss), val_accuracy, predicts, truths



def accuracy(target, output, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1. / batch_size))
    return res


''' ######################################################################## '''
''' ####################### BALANCED DATALOADER ############################ '''
''' ######################################################################## '''
# https://discuss.pytorch.org/t/how-does-weightedrandomsampler-work/8089

def create_sampler_weights(df, target_field, save_name, mu=1.0, save=True):
    """
    assign sample weights for each sample. rare sample have higher weights(linearly)
    refer to
    :param df:
    :param target_field:
    :param save_name:
    :param mu:
    :param save:
    :return:
    """
    label_list = df[target_field].tolist()
    import math
    import pickle
    import operator
    from functools import reduce
    from collections import Counter
    freq_count = dict(Counter(label_list))
    total = sum(freq_count.values())
    keys = freq_count.keys()
    assert sorted(list(keys)) == list(range(len(keys)))
    class_weight = dict()
    class_weight_log = dict()
    for key in range(len(keys)):
        score = total / float(freq_count[key])
        score_log = math.log(mu * total / float(freq_count[key]))
        class_weight[key] = round(score, 2) if score > 1.0 else round(1.0, 2)
        class_weight_log[key] = round(score_log, 2) if score_log > 1.0 else round(1.0, 2)

    rareness = [x[0] for x in sorted(freq_count.items(), key=operator.itemgetter(1))]

    weights = []
    sample_labels = label_list
    for label in sample_labels:
        for rare_label in rareness:
            if rare_label == label:
                weights.append(class_weight[rare_label])
                break

    assert len(weights) == len(label_list)
    if save:
        with open(save_name, 'wb') as f:
            pickle.dump(weights, f)
        print("%d weights saved into %s" % (len(label_list), save_name))
    else:
        return weights


''' ######################################################################## '''
''' ########################### LR FINDER ################################## '''
''' ######################################################################## '''


def findLR(model, optimizer, criterion, train_loader, final_value=10, init_value=1e-8, verbose=1):
    # https://medium.com/coinmonks/training-neural-networks-upto-10x-faster-3246d84caacd
    """
      findLR plots the graph for the optimum learning rates for the model with the
      corresponding dataset.
      The technique is quite simple. For one epoch,
      1. Start with a very small learning rate (around 1e-8) and increase the learning rate linearly.
      2. Plot the loss at each step of LR.
      3. Stop the learning rate finder when loss stops going down and starts increasing.

      A graph is created with the x axis having learning rates and the y axis
      having the losses.

      Arguments:
      1. model -  (torch.nn.Module) The deep learning pytorch network.
      2. optimizer: (torch.optim) The optimiser for the model eg: SGD,CrossEntropy etc
      3. criterion: (torch.nn) The loss function that is used for the model.
      4. trainloader: (torch.utils.data.DataLoader) The data loader that loads data in batches for input into model
      5. final_value: (float) Final value of learning rate
      6. init_value: (float) Starting learning rate.

      Returns:
       learning rates used and corresponding losses
    """
    model.train()  # setup model for training configuration

    num = len(train_loader) - 1  # total number of batches
    mult = (final_value / init_value) ** (1 / num)

    losses, lrs = [], []
    best_loss, avg_loss = 0., 0.
    beta = 0.98  # the value for smooth losses
    lr = init_value

    for batch_num, (inputs, targets) in enumerate(train_loader):

        if verbose == 1: print("Testint LR: {}".format(lr))
        optimizer.param_groups[0]['lr'] = lr
        batch_num += 1  # for non zero value
        inputs, targets = inputs.cuda(), targets.cuda()  # convert to cuda for GPU usage
        inputs = inputs.type(torch.float)
        optimizer.zero_grad()  # clear gradients
        outputs = model(inputs)  # forward pass
        loss = criterion(outputs, targets.long())  # compute loss

        # Compute the smoothed loss to create a clean graph
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)

        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        # append loss and learning rates for plotting
        lrs.append(math.log10(lr))
        losses.append(smoothed_loss)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break

        # backprop for next step
        loss.backward()
        optimizer.step()

        # update learning rate
        lr = mult * lr

    # plt.xlabel('Learning Rates')
    # plt.ylabel('Losses')
    # plt.plot(lrs,losses)
    # plt.show()
    return lrs, losses


''' ######################################################################## '''
''' #############################  CUTOUT ################################## '''
''' ######################################################################## '''


# https://arxiv.org/abs/1708.04552
# https://github.com/uoguelph-mlrg/Cutout


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img.cuda() * mask.cuda()

        return img


class BatchCutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, imgs):
        """
        Args:
            img (Tensor): Tensor image of size (Batch, C, H, W).
        Returns:
            Tensor: Images with n_holes of dimension length x length cut out of it.
        """

        h = imgs.size(2)
        w = imgs.size(3)

        outputs = torch.empty(imgs.shape)
        for index, img in enumerate(imgs):

            mask = np.ones((h, w), np.float32)

            for n in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            # imgs[index] = img.cuda() * mask.cuda()
            outputs[index] = img.cuda() * mask.cuda()

        # return imgs
        return outputs.cuda()


''' ######################################################################## '''
''' #############################  MIXUP ################################### '''
''' ######################################################################## '''


# mixup: BEYOND EMPIRICAL RISK MINIMIZATION: https://arxiv.org/abs/1710.09412
# https://github.com/facebookresearch/mixup-cifar10


### Ejemplo de uso
# inputs, targets_a, targets_b, lam = mixup_data(batch_data, batch_target, alpha_mixup)
# inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

# outputs = model(inputs)
# loss = mixup_criterion(loss_ce, outputs, targets_a, targets_b, lam)
# total_loss += loss.item()

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


''' ######################################################################## '''
''' ###########################  SAMPLE PAIRING ############################ '''
''' ######################################################################## '''


# https://arxiv.org/abs/1801.02929


def SamplePairing(img1, img2, alpha=0.5):
    """ Return the mixing of img1 with img2,
       tanking img1 alpha (over 1) and 1-alpha for img2
       (If alpha is 0.0, a copy of the first image is returned.
       If alpha is 1.0, a copy of the second image is returned.)
    """
    return PIL.Image.blend(img1, img2, alpha)
