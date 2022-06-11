import cv2
import numpy as np
import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', type=str, required=True, help='path to input video')
ap.add_argument('-s', '--steps', type=str, default=30, help='the number of frame from where it picks the sharpest,'
                                                            ' defaults to 30')
args = vars(ap.parse_args())


def main():
    img_path = args['input']  # '/media/thiago/HardStorage0/home/thiago/Storage/Photogrametry/IMG_9382.MOV'

    vid = cv2.VideoCapture(img_path)

    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counter = 0
    current_window_frame = 0
    current_img = None
    highest_mean = -1
    highest_counter = 0
    step = args['steps']
    ini_time = time.time()
    success, img = vid.read()

    while success:
        size = (int(img.shape[1]/2), int(img.shape[0]/2))
        img = cv2.resize(img, size)
        fft = np.fft.fft2(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        fft = np.fft.fftshift(fft)
        (h, w) = size
        (cX, cY) = (int(w / 2), int(h / 2))
        low_pass_win_size = 512
        fft[cY - low_pass_win_size:cY + low_pass_win_size, cX - low_pass_win_size:cX + low_pass_win_size] = 0
        fft_shift = np.fft.ifftshift(fft)
        recon = np.fft.ifft2(fft_shift)
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        win_counter = frame_counter // step

        if win_counter != current_window_frame:
            cv2.imwrite(f'frames/{win_counter:04}.png', current_img)
            highest_mean = -1
            current_window_frame = win_counter
            time_estimate = f'{(time.time() - ini_time) / frame_counter * total_frames/60:.2f}m remaining'
            print(f'Progress: {highest_counter/total_frames*100.0:.2f}% | {time_estimate}')

        if mean > highest_mean:
            highest_mean = mean
            current_img = img
            highest_counter = frame_counter

        success, img = vid.read()
        frame_counter += 1


if __name__ == '__main__':
    main()
