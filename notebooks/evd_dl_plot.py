import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import torchvision.transforms as transforms
import torch

def rotate_img(x, deg):
    """
    Rotate image (used to test uncertainty)
    """
    return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()


def rotating_image_classification(model, img, device, threshold=0.5):
    num_classes = 10
    Mdeg = 180
    Ndeg = int(Mdeg / 10) + 1
    ldeg = []
    lp = []
    lu = []
    classifications = []
    model.eval()
    scores = np.zeros((1, num_classes))
    rimgs = np.zeros((28, 28 * Ndeg))
    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
        nimg = rotate_img(img.numpy()[0], deg).reshape(28, 28)

        rimgs[:, i*28:(i+1)*28] = nimg
        trans = transforms.ToTensor()
        img_tensor = trans(nimg)

        logits, evidence, alpha, uncertainty = model(img_tensor.unsqueeze(0).to(device))
        _, preds = torch.max(logits, 1)
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)

        prob = prob.flatten()
        preds = preds.flatten()
        classifications.append(preds[0].item())
        lu.append(uncertainty.mean())


        scores += prob.detach().cpu().numpy() >= threshold
        ldeg.append(deg)
        lp.append(prob.tolist())

    labels = np.arange(10)[scores[0].astype(bool)]
    lp = np.array(lp)[:, labels]
    c = ["black", "blue", "red", "brown", "purple", "cyan"]
    marker = ["s", "^", "o"]*2
    labels = labels.tolist()
    fig = plt.figure(figsize=[6.2, 5]);
    fig, axs = plt.subplots(3, gridspec_kw={"height_ratios": [4, 2, 12]});

    for i in range(len(labels)):
        axs[2].plot(ldeg, lp[:, i], marker=marker[i], c=c[i])

    labels += ["uncertainty"]
    axs[2].plot(ldeg, lu, marker="<", c="red")

    print('Classifications:\n', classifications)

    axs[0].set_title("Rotated \"1\" Digit Classifications")
    axs[0].imshow(1 - rimgs, cmap="gray")
    axs[0].axis("off")

    empty_lst = []
    empty_lst.append(classifications)
    axs[1].table(cellText=empty_lst, bbox=[0, 1.2, 1, 1])
    axs[1].axis("off")

    axs[2].set_xlim([0, Mdeg])
    axs[2].set_ylim([0, 1])
    axs[2].set_xlabel("Rotation Degree")
    axs[2].set_ylabel("Classification Probability")
    axs[2].legend(labels)



