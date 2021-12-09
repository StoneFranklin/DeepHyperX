from thundersvm import *

import time

from utils import (
    metrics,
    sample_gt,
    build_dataset,
)
from datasets import get_dataset
from models import save_model

import argparse

parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)

parser.add_argument(
    "--cores",
    type=int,
    default=-1,
    help="Specify CUDA device (defaults to -1, which learns on CPU)",
)
parser.add_argument(
    "--memory",
    type=int,
    default=-1,
    help="Specify CUDA device (defaults to -1, which learns on CPU)",
)

args = parser.parse_args()

CORES = args.cores
MEMORY = args.memory
DATASET = "IndianPines"
FOLDER = "/mnt/Datasets/"
SAMPLE_PERCENTAGE = 0.8
SAMPLING_MODE = "random"
MODEL = "SVM"

img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)

train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)

N_BANDS = img.shape[-1]
N_CLASSES = len(LABEL_VALUES)

train1 = time.perf_counter()
print("------START TRAIN------")
X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
class_weight = "balanced"
print("------SVC------")
clf = SVC(class_weight=class_weight, n_jobs=CORES, max_mem_size=MEMORY)
print("------fit------")
clf.fit(X_train, y_train)
print("------STOP TRAIN------")
# Stop training
training_time = time.perf_counter() - train1
save_model(clf, MODEL, DATASET)
test_time1 = time.perf_counter()
# Start testing
prediction = clf.predict(img.reshape(-1, N_BANDS))
prediction = prediction.reshape(img.shape[:2])
testing_time = time.perf_counter() - test_time1

run_results = metrics(
    prediction,
    test_gt,
    ignored_labels=IGNORED_LABELS,
    n_classes=N_CLASSES,
    training_time=training_time,
    testing_time=testing_time
)

print("Accuracy: ")
print(run_results["Accuracy"])
print("Total Time: ")
print(run_results["TotalTime"])
