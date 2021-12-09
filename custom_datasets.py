from utils import open_file
import numpy as np

CUSTOM_DATASETS_CONFIG = {
    "DFC2018_HSI": {
        "img": "20170218_UH_CASI_S4_NAD83.hdr",
        "gt": "2018_IEEE_GRSS_DFC_GT_TR.tif",
        "download": False,
        "loader": lambda folder: dfc2018_loader(folder),
    }
}


def dfc2018_loader(folder):
    img = open_file(folder + "20170218_UH_CASI_S4_NAD83.hdr")[:, :, :] 
    print(img)
 
    gt = open_file(folder + "2018_IEEE_GRSS_DFC_GT_TR.tif")[:, :4172] 
    # gt = np.array(gt, dtype=np.uint8)
    gt = gt.astype("uint8")

    rgb_bands = (47, 31, 15)

    label_values = [
        "Unclassified",
        "Healthy grass",
        "Stressed grass",
        "Artificial turf",
        "Evergreen trees",
        "Deciduous trees",
        "Bare earth",
        "Water",
        "Residential buildings",
        "Non-residential buildings",
        "Roads",
        "Sidewalks",
        "Crosswalks",
        "Major thoroughfares",
        "Highways",
        "Railways",
        "Paved parking lots",
        "Unpaved parking lots",
        "Cars",
        "Trains",
        "Stadium seats",
    ]
    ignored_labels = [0]
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette
