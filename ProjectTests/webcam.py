import cv2
from speaker_tracker import SpeakerTracker
from matplotlib import pyplot as plt
import numpy as np
import time

def reject_outliers(data, m = 2):
    new_data = np.array(data)
    d = np.abs(data - np.median(new_data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0
    return new_data[s<m]

def plot_mouth_data():
    ratios = speaker_tracker.mouth.ratios
    plt.plot(ratios)
    plt.title('ratio')
    plt.savefig('plots/ratio')
    plt.figure()
    rolling_sample = []
    mean_diff = []
    max_list = []
    mean_list = []
    std = []
    mean_diff_list = []
    index = []
    mean_diff_mean = []
    for ratio in ratios:
        rolling_sample.insert(0, ratio)
        mean = np.mean(rolling_sample)
        mean_diff.insert(0, abs(ratio-mean))
        if len(rolling_sample)>12:
            rolling_sample.pop()
            mean_diff.pop()
        # remove outliers from rolling sample:
        outliers_removed = reject_outliers(rolling_sample)

        index.append(np.mean(outliers_removed)*np.std(outliers_removed))
        #mean_diff_mean.append(np.mean(mean_diff))
        #mean_diff_list.append(abs(ratio-mean))
        #max_list.append(max(rolling_sample))
        mean_list.append(np.mean(outliers_removed))
        std.append(np.std(outliers_removed))
    plot_vars = {'mean': mean_list, 'std': std, 'index': index, 'index': index}

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

