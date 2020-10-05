import cv2
from speaker_tracker import SpeakerTracker
from matplotlib import pyplot as plt
import numpy as np
import time

def plot_mouth_data():
    ratios = speaker_tracker.mouth.ratios
    plt.plot(ratios)
    plt.title('ratio')
    plt.savefig('plots/ratio')
    plt.figure()
    rolling_sample = []
    max_list = []
    min_list = []
    mean_list = []
    diff_list = []
    std = []
    index = []
    for ratio in ratios:
        rolling_sample.insert(0, ratio)
        if len(rolling_sample)>12:
            rolling_sample.pop()
        index.insert(0, max(rolling_sample)*np.mean(rolling_sample))
        max_list.insert(0,max(rolling_sample))
        min_list.insert(0, min(rolling_sample))
        mean_list.insert(0, np.mean(rolling_sample))
        diff_list.insert(0, max(rolling_sample) - min(rolling_sample))
        std.insert(0, np.std(rolling_sample))
    plot_vars = {'max': max_list, 'mean': mean_list, "diff": diff_list, 'std': std, 'index': index}

    for key, val in plot_vars.items():
        plt.plot(val)
        plt.title(key)
        plt.savefig('plots/' + key)
        plt.figure()

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

speaker_tracker = SpeakerTracker()

while True:
    time.sleep(0.2)
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    speaker_tracker.refresh(frame)

    cv2.imshow('Input', speaker_tracker.img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        # plot speaker data
        plot_mouth_data()
        break

