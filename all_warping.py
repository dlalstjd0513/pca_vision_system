import os
import numpy as np
import cv2

def warp(img):
    frame = img.copy()
    screenY, screenX = frame.shape[:2]

    resized_frame = cv2.resize(frame, dsize=(screenX // 12, screenY // 12), interpolation=cv2.INTER_AREA)

    # Increase the brightness (you can adjust the brightness level as needed)
    brightened_image = cv2.convertScaleAbs(resized_frame, alpha=2.0, beta=50)

    # Convert the brightened image to grayscale
    gray_image = cv2.cvtColor(brightened_image, cv2.COLOR_BGR2GRAY)

    # Perform histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)

    # Apply thresholding to create a binary image with inverted colors (black background, white object)
    ret, binary_image = cv2.threshold(equalized_image, 127, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological closing to fill empty parts of the object
    kernel = np.ones((10, 10), np.uint8)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Find contours in the closed image
    contours, hierarchy = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, cont in enumerate(contours):
        cnt_area = cv2.contourArea(cont)
        rect = cv2.minAreaRect(cont)
        box = cv2.boxPoints(rect)
        box = np.int0(box) * 12

    result_frame = warpImg(box, frame)

    return result_frame

def warpImg(box, frame):
    if len(box) == 4:
        sum_approx = box.sum(axis=1)
        diff_approx = np.diff(box, axis=1)
        topLeft = box[np.argmin(sum_approx)]
        bottomRight = box[np.argmax(sum_approx)]
        topRight = box[np.argmin(diff_approx)]
        bottomLeft = box[np.argmax(diff_approx)]

        pts1 = np.float32([topLeft, topRight, bottomLeft, bottomRight])

        width_bottom = abs(bottomRight[0] - bottomLeft[0])
        width_top = abs(topRight[0] - topLeft[0])
        height_right = abs(topRight[1] - bottomRight[1])
        height_left = abs(topLeft[1] - bottomLeft[1])
        pcb_width = max([width_bottom, width_top])
        pcb_height = max([height_right, height_left])

        pts2 = np.float32([[0, 0], [pcb_width, 0], [0, pcb_height], [pcb_width, pcb_height]])
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        warp_frame = cv2.warpPerspective(frame, mtrx, (pcb_width, pcb_height), flags=cv2.INTER_CUBIC)

        return warp_frame

if __name__ == '__main__':
    print('ProGramStart')
    # Get list of all BMP files in the current directory
    bmp_files = [f for f in os.listdir('.') if f.lower().endswith('.bmp')]
    
    # Directory to save processed images
    output_dir = 'warp'
    os.makedirs(output_dir, exist_ok=True)
    
    for bmp_file in bmp_files:
        print(f'Processing {bmp_file}')
        frame = cv2.imread(bmp_file)
        if frame is not None:
            warp_result_frame = warp(frame)
            
            # Save the processed image in the specified directory
            output_filename = os.path.join(output_dir, f'processed_{bmp_file}')
            cv2.imwrite(output_filename, warp_result_frame)
            print(f'Saved {output_filename}')
        else:
            print(f'Failed to read {bmp_file}')

