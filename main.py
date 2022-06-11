import cv2
import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', type=str, required=True, help='path to input video')
ap.add_argument('-s', '--steps', type=int, default=30, help='the number of frame from where it picks the sharpest,'
                                                            ' defaults to 30')
ap.add_argument('-f', '--scale_factor', type=int, default=8, help='how many times image will be halved')
args = vars(ap.parse_args())


def main():
    img_path = args['input']

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
    csv = 'Filename,id,frame\n'
    scale_factor = args['scale_factor']

    while success:
        size = (int(img.shape[1]/scale_factor), int(img.shape[0]/scale_factor))
        img = cv2.resize(img, size)
        mean = cv2.Laplacian(img, cv2.CV_64F).var()
        win_counter = frame_counter // step

        if win_counter != current_window_frame:
            file_name = f'frames/{win_counter:03}.png'
            cv2.imwrite(file_name, current_img)
            highest_mean = -1
            current_window_frame = win_counter
            time_estimate = f'{(time.time() - ini_time) / frame_counter * total_frames/60:.2f}m remaining'
            print(f'Progress: {highest_counter/total_frames*100.0:.2f}% | {time_estimate}')
            csv = f'{csv}{file_name},{win_counter},{highest_counter}\n'
        if mean > highest_mean:
            highest_mean = mean
            current_img = img
            highest_counter = frame_counter

        success, img = vid.read()
        frame_counter += 1
    with open('log.csv', 'w') as file:
        file.write(csv)


if __name__ == '__main__':
    main()
