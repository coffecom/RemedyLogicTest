import torch
from torchvision.transforms import functional 
from pathlib import Path
import h5py
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

def resize_image_and_mask(image, mask, size = picture_size):
  image = functional.resize(image, size)
  mask = functional.resize(mask, size, transforms.InterpolationMode.NEAREST)
  return image, mask

def visualize_prediction(image, prediction):
    summs_along_channels = torch.sum(prediction, dim=[1,2])
    armax = torch.argmax(summs_along_channels).cpu().numpy()
    pred_class = labels[armax + 1]
    mask_transperent = prediction[armax].numpy().astype(np.float)
    mask_transperent[np.where(mask_transperent < 0.9)] = np.nan

    plt.title("PREDICTED CLASS:" + pred_class)
    plt.imshow(image)
    plt.imshow(mask_transperent, cmap='jet', alpha = 0.7)

    plt.savefig("result.png")

def prepare_image(file_path):
    f = h5py.File(file_path, 'r')
    image = np.array(f['cjdata']['image'])

    image = torch.Tensor([[image]])
    print("SHAPE:", image.shape)
    image = functional.resize(image, picture_size)

    return image.float()

def predict(file_path):
    image = prepare_image(file_path)
    res = model.predict(image) 
    visualize_prediction(image[0][0].numpy(), res[0]) # TODO add ground truth visualisation
    

# predict(Path("D:\\programming\\remedylogic\\brainTumorDataPublic\\brainTumorDataPublic_767-1532\\767.mat"))