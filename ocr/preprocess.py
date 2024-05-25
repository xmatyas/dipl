import cv2 as cv
import numpy as np
from PIL import Image
from imutils.perspective import four_point_transform
import yaml
import pickle
import os

from ocr.utils import show_image, load_and_convert, resize_with_ratio, load_config

def prepare_image(image):
    original_image_size = load_and_convert(image, grayscale=True).shape
    config = load_config('config.yaml')['preprocessing']
    
    # Preprocess the passed image/path
    image = load_and_convert(image, grayscale=True)
    image = image.copy()
    
    if config['resize']['enabled']:
        image = resize_with_ratio(image, config['resize']['height'], return_pil= config['resize']['return_pil'])
    
    # Blurring
    def apply_blur(image, blur_type=0, kernel_size=5):
        if blur_type == 0:
            blurred_image = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif blur_type == 1:
            blurred_image = cv.medianBlur(image, kernel_size)
        elif blur_type == 2:
            blurred_image = cv.bilateralFilter(image, kernel_size, 75, 75)
        return blurred_image
    
    # Closing and opening operations
    def morphological_transformations(image, kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        for i in range(3):
            image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=3)
            image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=3)
        return image
    
    # Thresholding
    def apply_threshold(image, adaptive = False):
        if adaptive == True:
            thresholded_image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        else:
            otsu_threshold, thresholded_image = cv.threshold(image, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
        return thresholded_image
    
    # Canny
    def apply_canny(image, low_threshold=100, high_threshold=200):
        edges = cv.Canny(image, low_threshold, high_threshold)
        return edges
    
    # Erode and dilate
    def erode_and_dilate(image, kernel_size=5, erode_iterations=1, dilate_iterations=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_image = cv.dilate(image, kernel, iterations=dilate_iterations)
        eroded_image = cv.erode(dilated_image, kernel, iterations=erode_iterations)
        return eroded_image
    
    if config['blurring']['enabled']:
        image = apply_blur(image, blur_type=config['blurring']['type'], kernel_size=config['blurring']['kernel_size'])
    
    if config['thresholding']['enabled']:
        image = apply_threshold(image, adaptive=config['thresholding']['adaptive'])

    if config['morphological_transformations']['enabled']:
        image = morphological_transformations(image, kernel_size=config['morphological_transformations']['kernel_size'])
    
    if config['erode_and_dilate']['enabled']:
        image = erode_and_dilate(image, kernel_size=config['erode_and_dilate']['kernel_size'], erode_iterations=config['erode_and_dilate']['erode_iterations'], dilate_iterations=config['erode_and_dilate']['dilate_iterations'])
    
    if config['canny_edge_detection']['enabled']:
        image = apply_canny(image, low_threshold=config['canny_edge_detection']['low_threshold'], high_threshold=config['canny_edge_detection']['high_threshold'])
    
    image = resize_with_ratio(image, width=original_image_size[1], return_pil=False)
    return image

def transform_image(original_image, preprocessed_image):
    # Detect if the document contour is found
    def detect_contour(image):
        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        document_contour = None
        max_area = image.shape[0] * image.shape[1] * 0.75
        
        for contour in contours:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4 and cv.isContourConvex(approx):
                area = cv.contourArea(approx)
                if area > max_area:
                    max_area = area
                    document_contour = approx
        return document_contour
    
    def perspective_transform(image, document_contour):
        document_contour = document_contour.reshape(4, 2)
        document_contour = document_contour.astype(np.float32)
        image = four_point_transform(image, document_contour)
        return image
    
    def detect_features(image, template):
        # Init SIFT detector
        sift = cv.SIFT_create()
        # Keypoint detection
        keypoints_image, descriptors_image = sift.detectAndCompute(image, None)
        keypoints_template, descriptors_template = sift.detectAndCompute(template, None)
        
        # FLANN matcher
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        
        # Match the descriptors
        matches = flann.knnMatch(descriptors_template, descriptors_image, k=2)
        
        # Store all the good matches as per Lowe's ratio test
        good_matches = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return keypoints_image, keypoints_template, good_matches
    
    def align_document(image, template, keypoints_image, keypoints_template, matches):
        # Extract location of good matches
        points1 = np.float32([keypoints_template[m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints_image[m.trainIdx].pt for m in matches])
        
        # Find the homography matrix
        matrix, mask = cv.findHomography(points2, points1, cv.RANSAC, 5.0)
        aligned_image = cv.warpPerspective(image, matrix, (template.shape[1], template.shape[0]))
        return aligned_image
    
    def extract_corner(image, corner='top_left', segment_size=100, ingore_flip=False):
        if corner == 'top_left':
            corner_segment = image[:segment_size, :segment_size]
        elif corner == 'top_right':
            corner_segment = image[:segment_size, -segment_size:]
            if not ingore_flip:
                corner_segment = cv.flip(corner_segment, 1)
        elif corner == 'bottom_left':
            corner_segment = image[-segment_size:, :segment_size]
            if not ingore_flip:
                corner_segment = cv.flip(corner_segment, 0)
        elif corner == 'bottom_right':
            corner_segment = image[-segment_size:, -segment_size:]
            if not ingore_flip:
                corner_segment = cv.flip(corner_segment, -1)
        else:
            raise ValueError("Invalid corner specified")
        
        _, thresh_segment = cv.threshold(corner_segment, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        return thresh_segment
    
    def check_mark(corner_image, corner, line_length=20, tolerance=0.8):
        
        rows, cols = corner_image.shape
        white_pixel_threshold = 255 * tolerance
        
        for y in range(1, rows - line_length - 1):  # Start from 1 and end a bit earlier to check surroundings
            for x in range(1, cols - line_length - 1):
                horizontal_line = corner_image[y, x:x + line_length]
                vertical_line = corner_image[y:y + line_length, x]

                # Check if the horizontal and vertical lines are predominantly white
                if (np.mean(horizontal_line) > white_pixel_threshold) and (np.mean(vertical_line) > white_pixel_threshold):
                    # Check surrounding pixels to ensure it's not a uniformly white area
                    if is_distinct_shape(corner_image, x, y, line_length):
                        crop_point = (x + int(line_length/3), y + int(line_length/3))
                        # Adjust the crop point based on the corner, since the image is mirror flipped
                        if corner == 'top_right':
                            crop_point = (cols - crop_point[0] - 1, crop_point[1])
                        elif corner == 'bottom_left':
                            crop_point = (crop_point[0], rows - crop_point[1] - 1)
                        elif corner == 'bottom_right':
                            crop_point = (cols - crop_point[0] - 1, rows - crop_point[1] - 1)
                        return True, crop_point
        return False, None
    
    def is_distinct_shape(image, x, y, line_length):
        # Check the pixels just outside the L-shape
        if np.mean(image[y - 1, x:x + line_length]) < 255 * 0.5 and np.mean(image[y:y + line_length, x - 1]) < 255 * 0.5:
            return True
        return False

    def detect_marks(image, segment_size=100, line_length=20, tolerance=0.8):
        #image = load_and_convert(image, grayscale=True)
        corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        results = {}

        for corner in corners:
            corner_image = extract_corner(image, corner=corner, segment_size=segment_size)
            detected, crop_point = check_mark(corner_image, corner=corner, line_length=line_length, tolerance=tolerance)
            # Debugging
            #corner_image_unmirrored = extract_corner(image, corner=corner, segment_size=segment_size, ingore_flip=True)
            #corner_image_unmirrored = cv.cvtColor(corner_image_unmirrored, cv.COLOR_GRAY2BGR)
            #print(f"{corner} detected: {detected}, crop point: {crop_point}")
            #cv.circle(corner_image_unmirrored, crop_point, 5, (0, 0, 255), -1)
            #show_image(corner_image_unmirrored)
            #cv.imwrite(f'corner_{corner}.jpg', corner_image_unmirrored)
            results[corner] = detected, crop_point        
        
        return results
    
    def convert_local_to_global_points(image, results, segment_size=100):
        height, width = image.shape
        pts = []
        
        for result in results:
            # Check if the mark was detected
            if results[result][0]:
                x, y = results[result][1]
                if result == 'top_left':
                    pts.append((x, y))
                elif result == 'top_right':
                    pts.append((width - segment_size + x, y))
                elif result == 'bottom_left':
                    pts.append((x, height - segment_size + y))
                elif result == 'bottom_right':
                    pts.append((width - segment_size + x, height - segment_size + y))
            else:
                # All four corners must be detected
                return None      
        return np.array(pts)
    
    def crop_image(image, results, segment_size=100):
        pts = convert_local_to_global_points(image, results, segment_size=segment_size)
        return four_point_transform(image, pts)
        
    # Create copies of the images, so the original images are not modified
    image = original_image.copy()
    gray = preprocessed_image.copy()
    
    # Detect the document contour
    document_contour = detect_contour(gray)
    
    # Transform the image, if contour is found (in case of photo of a document and not a scan)
    if document_contour is not None:
        image = perspective_transform(image, document_contour)
    else:
        print("Document contour not found")
    
    # Load the config
    config = load_config('config.yaml')['transformation']
    # Load and prepare the template and image in the context of the ocr folder)
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    current_template_path = os.path.join(curr_dir, config['template']['path'])
    template = load_and_convert(current_template_path, grayscale=True)
    # Convert the image to the same size as the template
    comparison_width, _ = template.shape
    image = resize_with_ratio(image, width=comparison_width, return_pil=False)
    # Detect the features and align the document
    kp1, kp2, matches = detect_features(image, template)
    aligned_image = align_document(image, template, kp1, kp2, matches)
    # Resize to static size, this proved to work with default segment size. Different sizes may require adjustments
    aligned_image = resize_with_ratio(aligned_image, width=1000, return_pil=False)
    
    # Detect the marks in the static size image
    result = detect_marks(aligned_image, tolerance=0.3)
    aligned_image = crop_image(aligned_image, result)
    
    # Revert the image to the original size and return
    aligned_image = resize_with_ratio(aligned_image, width=original_image.shape[1], return_pil=False)
    return aligned_image
    
def parse_image(image):
    prepared_image = prepare_image(image)
    original_image = load_and_convert(image, grayscale=False)
    transformed_image = transform_image(original_image, prepared_image)
    return transformed_image

def extract(image):
    # # Load the config
    # def load_config(filename='block_config.yaml'):
    #     with open(filename, 'r') as file:
    #         return yaml.safe_load(file)
    
    def extract_sequence(image, sequence_info, block_width, block_height, spacing, lightness_threshold, target_size=(28,28)):
        num_of_blocks = sequence_info['blocks']
        start_x = sequence_info['x']
        start_y = sequence_info['y']
        sequence = []
        for block in range(num_of_blocks):
            block = extract_block(image, block, start_x, start_y, block_width, block_height, spacing, target_size=(28,28))
            if np.mean(block) > lightness_threshold:
                sequence.append(block)
        return sequence
    
    def extract_block(image, block, start_x, start_y, block_width, block_height, spacing, target_size=(28,28)):
        current_x = start_x + block * (block_width + spacing)
        current_y = start_y # The current y should stay the same for the whole sequence (if the alignment is correct)
        block = image[current_y:current_y + block_height, current_x:current_x + block_width] # Extract the block
        resized_block = cv.resize(block, target_size, interpolation=cv.INTER_AREA) # Resize the block to 28x28
        thresholded_block = cv.adaptiveThreshold(resized_block, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 10) # Threshold the block
        reshaped_block = thresholded_block.reshape(1, 28, 28) # Reshape the block to 1D
        normalized_block = reshaped_block / 255.0 # Normalize the block
        return normalized_block
        
        
    config = load_config('block_config.yaml')
    BLOCK_WIDTH = config['settings']['block_width']
    BLOCK_HEIGHT = config['settings']['block_height']
    SPACING = config['settings']['spacing']
    LIGHTNESS_THRESHOLD = config['settings']['lightness_threshold']
    image = resize_with_ratio(image, width=1000, grayscale=False, return_pil=False)
    
    image_blocks = []
    
    for sequence_info in config['sequences']:
        seq = extract_sequence(image, sequence_info, BLOCK_WIDTH, BLOCK_HEIGHT, SPACING, LIGHTNESS_THRESHOLD, target_size=(28,28))
        image_blocks.append(seq)
    
    return image_blocks

def process_image(image, output = None):
    image = parse_image(image)
    image_blocks = extract(image)
    if output is not None:
        with open('./tmp/' + output + '.pkl', 'wb') as file:
            pickle.dump(image_blocks, file)
    return image_blocks

if __name__ == '__main__':
    process_image('../input/template/sample101.jpg')