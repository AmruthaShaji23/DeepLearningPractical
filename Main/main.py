import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from PIL import Image
import os

hebrew_class_map = {
    "alef": "א",
    "ayin": "ע",
    "bet": "ב",
    "dalet": "ד",
    "gimel": "ג",
    "he": "ה",
    "het": "ח",
    "kaf": "כ",
    "kaf_final": "ך",
    "lamed": "ל",
    "mem": "מ",
    "mem-medial": "מ",
    "nun_final": "ן",
    "nun_medial": "נ",
    "pe": "פ",
    "pe_final": "ף",
    "qof": "ק",
    "resh": "ר",
    "samekh": "ס",
    "shin": "ש",
    "taw": "ת",
    "tet": "ט",
    "tsadi_final": "ץ",
    "tsadi_medial": "צ",
    "waw": "ו",
    "yod": "י",
    "zayin": "ז",
}


def crop_and_save_lines(corrected_image, line_peaks, image_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    line_images = []

    height = corrected_image.height
    top = 0
    for i, bottom in enumerate(line_peaks + [height]):
        line_img = corrected_image.crop((0, top, corrected_image.width, bottom))
        line_filename = f"{image_name}_line_{i}.jpg"
        line_path = os.path.join(save_dir, line_filename)
        line_img.save(line_path)
        line_images.append(line_path)
        top = bottom

    return line_images


def transcribe_line_image(image_path, model, class_names_map):
    img = Image.open(image_path).convert("RGB")
    results = model.predict(source=img, conf=0.25, save=False)[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return ""

    characters = []
    for box in boxes:
        cls_id = int(box.cls.item())
        x_center = box.xywh[0][0].item()
        class_name = model.model.names[cls_id]
        char = class_names_map.get(class_name, "")  # Convert to Unicode
        characters.append((x_center, char))

    # Sort characters left to right
    characters.sort(key=lambda x: x[0])
    transcription = "".join(
        [char for _, char in sorted(characters, key=lambda x: -x[0])]
    )
    # No space between letters
    return transcription


def find_optimal_cuts(projection, peaks, min_distance=30, min_height_ratio=1):
    valleys = []

    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        segment = projection[start:end]

        # find local minima
        candidates, _ = find_peaks(-segment, distance=min_distance)

        if candidates.size > 0:
            # deepest
            deepest = candidates[np.argmin(segment[candidates])]
            valleys.append(start + deepest)
        else:
            # midpoint
            valleys.append(start + len(segment) // 2)

    return valleys


def detect_skew_with_hough(image):
    # to array
    img_array = np.array(image)

    # Apply adaptive thresholding to clean up the image
    img_array = cv2.adaptiveThreshold(
        img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    # connect characters
    kernel = np.ones((3, 3), np.uint8)
    img_array = cv2.dilate(img_array, kernel, iterations=1)

    # find edges
    edges = cv2.Canny(img_array, 50, 150, apertureSize=3)

    # hough transform to find lines
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=min(image.size) // 3,  # Dynamic minimum length
        maxLineGap=20,
    )

    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # find angles of lines
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # filter out vertical lines and extreme angles
            if abs(angle) < 20:
                # weigh the angle by the length of the line
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angles.extend(
                    [angle] * int(length / 10)
                )  # Add multiple times based on length

    if not angles:
        return 0

    # sort angles and calculate weighted median
    angles_sorted = np.sort(angles)
    weights = np.arange(1, len(angles_sorted) + 1)
    weighted_median = angles_sorted[np.argmax(np.cumsum(weights) >= sum(weights) / 2)]

    return weighted_median


def horizontalProjectionAndSkewAlignment(image, skew_range=15):
    # detect skew using Hough transform
    skew_angle = detect_skew_with_hough(image)
    print(f"Detected skew angle: {skew_angle:.2f}°")
    best_score = -np.inf
    best_angle = 0
    best_rotated = image

    # adjust the skew angle to find the best alignment
    for angle in np.linspace(skew_angle - 3, skew_angle + 3, 7):
        print(f"Testing angle: {angle:.2f}°")
        rotated = image.rotate(angle, fillcolor="white")
        projection = np.sum(rotated, axis=1)
        smoothed = gaussian_filter1d(projection, sigma=13)
        peaks, props = find_peaks(smoothed, prominence=700, distance=20)

        if len(peaks) < 2:  # Not enough lines
            continue

        # calculate spacing ratio and average prominence
        spacing = np.diff(peaks)
        spacing_ratio = np.std(spacing) / np.mean(spacing)
        avg_prominence = np.mean(props["prominences"])
        score = avg_prominence * (1 - spacing_ratio)

        if score > best_score:
            best_score = score
            best_angle = angle
            best_rotated = rotated

    # project, smooth, and find optimal line cuts
    projection = np.sum(best_rotated, axis=1)
    smoothed = gaussian_filter1d(projection, sigma=18)
    inverted = np.max(smoothed) - smoothed
    peaks, _ = find_peaks(inverted, prominence=700, distance=20)
    valleys = find_optimal_cuts(smoothed, peaks)
    print(f"Detected lines: {len(valleys) + 1}")
    print(f"Best angle: {best_angle:.2f}°")
    """# visualize the results
    plt.figure(figsize=(10, 4))
    plt.plot(smoothed, label=f"Skew: {best_angle:.1f}°")
    plt.scatter(peaks, smoothed[peaks], color="red", label="Text Lines")
    plt.scatter(valleys, smoothed[valleys], color="blue", marker="o", label="Line Cuts")
    plt.title(f"Image Projection")
    plt.xlabel("Row Index")
    plt.ylabel("Pixel Sum")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()"""

    return best_angle, valleys, best_rotated


class_names = [
    "alef",
    "ayin",
    "bet",
    "dalet",
    "gimel",
    "he",
    "het",
    "kaf",
    "kaf_final",
    "lamed",
    "mem",
    "mem-medial",
    "nun_final",
    "nun_medial",
    "pe",
    "pe_final",
    "qof",
    "resh",
    "samekh",
    "shin",
    "taw",
    "tet",
    "tsadi_final",
    "tsadi_medial",
    "waw",
    "yod",
    "zayin",
]


def process_and_combine_all(
    input_dir, output_dir, model_path, class_names, line_prefix, k
):
    """
    Processes line images with YOLO and combines both visual predictions and label files

    Args:
        input_dir: Directory containing line images
        output_dir: Where to save combined results
        model_path: Path to YOLO model weights
        class_names: List of class names in order
        line_prefix: Prefix to identify line images
        k: Identifier number for output files
    """
    # Initialize model
    model = YOLO(model_path)
    model.model.names = class_names

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Get all line images
    line_images = sorted(
        [f for f in os.listdir(input_dir) if f.startswith(line_prefix)]
    )

    line_results = []
    all_labels = []
    current_y_offset = 0  # Track vertical position for combining lines

    for line_img in line_images:
        img_path = os.path.join(input_dir, line_img)
        img = Image.open(img_path).convert("RGB")
        img_height = img.height

        # Run prediction
        results = model.predict(
            source=img,
            conf=0.25,
            save=False,
            line_width=1,
            show_labels=True,
            show_conf=True,
            iou=0.4,
        )

        # Get annotated image
        annotated_img = results[0].plot(line_width=1, font_size=10)
        line_results.append(Image.fromarray(annotated_img[..., ::-1]))  # BGR to RGB

        # Process and store labels with y-offset adjustment
        for box in results[0].boxes:
            cls = int(box.cls.item())
            conf = box.conf.item()
            x_center, y_center, width, height = box.xywhn[0].tolist()

            # Adjust y-coordinate for combined image
            adjusted_y_center = (current_y_offset + y_center * img_height) / (
                current_y_offset + img_height
            )
            adjusted_height = height * img_height / (current_y_offset + img_height)

            all_labels.append(
                f"{cls} {x_center:.6f} {adjusted_y_center:.6f} {width:.6f} {adjusted_height:.6f} {conf:.6f}\n"
            )

        current_y_offset += img_height

    # Save combined outputs if results exist
    if line_results:
        # Create combined image
        widths, heights = zip(*(i.size for i in line_results))
        combined_img = Image.new("RGB", (max(widths), sum(heights)))
        y_offset = 0
        for img in line_results:
            combined_img.paste(img, (0, y_offset))
            y_offset += img.height

        # Save combined image
        img_output_path = os.path.join(output_dir, f"combined_predictions_{k}.jpg")
        combined_img.save(img_output_path)

        # Save combined labels
        # txt_output_path = os.path.join(output_dir, f"combined_labels_{k}.txt")
        """with open(txt_output_path, "w") as f:
            f.writelines(all_labels)"""

        print(
            f"Saved combined results to:\n"
            f"- Image: {img_output_path}\n"
            # f"- Labels: {txt_output_path}"
        )

        return img_output_path, None
    else:
        print(f"No line images found for prefix: {line_prefix}")
        return None, None


def main():
    folder = "test_images"
    output_txt_dir = "results"
    os.makedirs(output_txt_dir, exist_ok=True)
    model_path = "best.pt"
    model = YOLO(model_path)
    model.model.names = class_names

    binarized_images = [f for f in os.listdir(folder)]
    print(f"Found {len(binarized_images)} binarized images.")

    for image_file in binarized_images:
        image_path = os.path.join(folder, image_file)
        image = Image.open(image_path)
        base_name = os.path.splitext(image_file)[0]
        print(f"Processing {image_file}...")
        # Skew correction and line detection
        skew, line_peaks, corrected_image = horizontalProjectionAndSkewAlignment(image)
        print(
            f"Processing {image_file} | Skew: {skew:.2f}° | Lines: {len(line_peaks)+1}"
        )

        # Crop and save lines
        line_dir = f"temp_lines/{base_name}"
        line_image_paths = crop_and_save_lines(
            corrected_image, line_peaks, base_name, line_dir
        )

        # Transcribe each line
        transcriptions = []

        for line_img_path in line_image_paths:
            transcription = transcribe_line_image(
                line_img_path, model, hebrew_class_map
            )
            transcriptions.append(transcription)

        # Save transcriptions to txt
        txt_path = os.path.join(output_txt_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for line in transcriptions:
                f.write(line + "\n")
        print(f"Saved transcription to {txt_path}")

        # Generate and save combined annotated image
        process_and_combine_all(
            input_dir=line_dir,
            output_dir=output_txt_dir,
            model_path=model_path,
            class_names=class_names,
            line_prefix=base_name,
            k=base_name,
        )


if __name__ == "__main__":
    main()
