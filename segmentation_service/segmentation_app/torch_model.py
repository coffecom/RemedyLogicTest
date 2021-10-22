import torch
from torchvision.transforms import functional 
from torchvision import transforms
from pathlib import Path
import h5py
import io, base64
import matplotlib.pyplot as plt
import numpy as np

file_directory = Path(__file__).parent.resolve()
model = torch.load(file_directory / "models/FPN_multilabel_best_model.pth", map_location = torch.device('cpu')) # TODO change model

labels = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary tumor"
}
picture_size = 256

def visualize_prediction(image, prediction):
    summs_along_channels = torch.sum(prediction, dim=[1,2])
    armax = torch.argmax(summs_along_channels).cpu().numpy()
    pred_label = labels[armax + 1]
    mask_transperent = prediction[armax].numpy().astype(np.float)
    mask_transperent[np.where(mask_transperent < 0.9)] = np.nan

    plt.title("PREDICTED LABEL:" + pred_label)
    plt.imshow(image)
    plt.imshow(mask_transperent, cmap='jet', alpha = 0.7)

    plt.savefig("result.png")

def transperent_mask(mask):
    mask_transperent = mask.astype(np.float)
    mask_transperent[np.where(mask_transperent < 0.9)] = np.nan
    return mask_transperent

def visualize_gt_pred(image, prediction, original_image, mask, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    gt_label = labels[label]
    summs_along_channels = torch.sum(prediction, dim=[1,2])
    armax = torch.argmax(summs_along_channels).cpu().numpy()
    pred_label = labels[armax + 1]

    ax1.set_title("Ground truth label:" + gt_label)
    ax1.imshow(original_image)
    ax1.imshow(transperent_mask(mask), cmap='jet', alpha = 0.7)
    ax1.legend()

    ax2.set_title("Predicted label:" + pred_label)
    ax2.imshow(image)
    ax2.imshow(transperent_mask(prediction[armax].numpy()), cmap='jet', alpha = 0.7)
    ax2.legend()

def prepare_image(raw_image):
    image = torch.Tensor([[raw_image]])
    image = functional.resize(image, picture_size)

    return image.float()

def predict(file_path):
    f = h5py.File(file_path, 'r')
    image = np.array(f['cjdata']['image'])

    prepared_image = prepare_image(image)
    res = model.predict(prepared_image) 

    if ('tumorMask' in f['cjdata'] and 'label' in f['cjdata']):
        mask = np.array(f['cjdata']['tumorMask'])
        label = f['cjdata']['label']
        visualize_gt_pred(prepared_image[0][0], res[0], image, mask, label[0][0])
    else:
        visualize_prediction(prepared_image[0][0], res[0]) 
    
    flike = io.BytesIO()
    plt.savefig(flike)
    b64 = base64.b64encode(flike.getvalue()).decode()
    return b64