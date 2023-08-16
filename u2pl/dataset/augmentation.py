import numpy as np
import torch

def generate_cutout_mask(img_size, ratio=2):
    cutout_area = img_size[0] * img_size[1] / ratio

    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.long()


def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)  # all unique labels
    labels_select = labels[torch.randperm(len(labels))][
        : len(labels) // 2
    ]  # randomly select half of labels

    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.float()


def generate_unsup_data(data, target, logits, mode="cutout"):
    batch_size, _, im_h, im_w = data.shape
    device = data.device

    new_data = []
    new_target = []
    new_logits = []
    for i in range(batch_size):
        if mode == "cutout":
            mix_mask = generate_cutout_mask([im_h, im_w], ratio=2).to(device)
            target[i][(1 - mix_mask).bool()] = 255

            new_data.append((data[i] * mix_mask).unsqueeze(0))
            new_target.append(target[i].unsqueeze(0))
            new_logits.append((logits[i] * mix_mask).unsqueeze(0))
            continue

        if mode == "cutmix":
            mix_mask = generate_cutout_mask([im_h, im_w]).to(device)
        if mode == "classmix":
            mix_mask = generate_class_mask(target[i]).to(device)

        new_data.append(
            (
                data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_target.append(
            (
                target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_logits.append(
            (
                logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )

    new_data, new_target, new_logits = (
        torch.cat(new_data),
        torch.cat(new_target),
        torch.cat(new_logits),
    )
    return new_data, new_target.long(), new_logits
