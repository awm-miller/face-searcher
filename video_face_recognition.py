import cv2
import face_recognition
import os
import json
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
from datetime import datetime
import traceback
import sys
import argparse
import time
import psutil
import torch  # To check for CUDA availability
import random
from PIL import Image, ImageEnhance

class VideoFaceRecognizer:
    def __init__(self, faces_library_path: str, detection_model: str = "hog"):
        """
        Initialize the face recognizer with a library of known faces.
        
        Args:
            faces_library_path (str): Path to the directory containing known face images
            detection_model (str): Face detection model to use - either "hog" (faster) or "cnn" (more accurate, requires GPU)
        """
        # Print system info
        print("\n=== System Information ===")
        print(f"CPU Count: {psutil.cpu_count()} ({psutil.cpu_count(logical=False)} physical)")
        print(f"Available Memory: {psutil.virtual_memory().available / (1024*1024*1024):.1f}GB")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("========================\n")
        
        self.faces_library_path = Path(faces_library_path)
        self.cache_file = self.faces_library_path / "face_encodings_cache.json"
        self.known_face_encodings = []
        self.known_face_names = []
        self.detection_model = detection_model
        if detection_model not in ["hog", "cnn"]:
            raise ValueError("detection_model must be either 'hog' or 'cnn'")
            
        if detection_model == "cnn" and not torch.cuda.is_available():
            print("\n⚠️ WARNING: Using CNN model without CUDA support may be very slow!")
            print("Consider using --model hog for better performance on CPU\n")
            
        self.load_known_faces()

    def _get_file_metadata(self, image_path: Path) -> dict:
        """Get file metadata for caching purposes."""
        stats = image_path.stat()
        return {
            "modified_time": stats.st_mtime,
            "size": stats.st_size
        }

    def _load_cache(self) -> Dict:
        """Load the face encodings cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                return cache_data
            except json.JSONDecodeError:
                print("Cache file corrupted, will rebuild cache")
        return {}

    def _save_cache(self, cache_data: Dict) -> None:
        """Save the face encodings cache to file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_cache = {}
        for path, data in cache_data.items():
            encoding = data["encoding"]
            # Convert to list if it's a numpy array
            if encoding is not None and hasattr(encoding, 'tolist'):
                encoding = encoding.tolist()
            
            serializable_cache[path] = {
                "encoding": encoding,
                "metadata": data["metadata"]
            }
        
        with open(self.cache_file, 'w') as f:
            json.dump(serializable_cache, f)

    def _augment_face_image(self, image_path: Path, output_dir: Path) -> List[Path]:
        """
        Create augmented versions of a face image with various transformations.
        
        Args:
            image_path: Path to the original image
            output_dir: Directory to save augmented images
            
        Returns:
            List of paths to augmented images
        """
        augmented_paths = []
        try:
            # Open image with PIL for better augmentation support
            img = Image.open(image_path)
            base_name = image_path.stem
            
            # Store original size for consistent output
            original_size = img.size
            
            # 1. Horizontal Flip
            flip_path = output_dir / f"{base_name}_flip.jpg"
            if not flip_path.exists():
                img.transpose(Image.FLIP_LEFT_RIGHT).save(flip_path, quality=95)
                augmented_paths.append(flip_path)
            
            # 2. Slight Rotations (-10, -5, 5, 10 degrees)
            for angle in [-10, -5, 5, 10]:
                rot_path = output_dir / f"{base_name}_rot{angle}.jpg"
                if not rot_path.exists():
                    rotated = img.rotate(angle, expand=True)
                    # Ensure same size as original
                    rotated = rotated.resize(original_size)
                    rotated.save(rot_path, quality=95)
                    augmented_paths.append(rot_path)
            
            # 3. Brightness Variations
            enhancer = ImageEnhance.Brightness(img)
            for factor in [0.8, 1.2]:  # Darker and brighter
                bright_path = output_dir / f"{base_name}_bright{factor:.1f}.jpg"
                if not bright_path.exists():
                    enhancer.enhance(factor).save(bright_path, quality=95)
                    augmented_paths.append(bright_path)
            
            # 4. Contrast Variations
            enhancer = ImageEnhance.Contrast(img)
            for factor in [0.8, 1.2]:  # Less and more contrast
                contrast_path = output_dir / f"{base_name}_contrast{factor:.1f}.jpg"
                if not contrast_path.exists():
                    enhancer.enhance(factor).save(contrast_path, quality=95)
                    augmented_paths.append(contrast_path)
            
            # 5. Small Zoom (crop and resize back)
            zoom_factor = 0.9  # 10% zoom
            width, height = img.size
            new_width = int(width * zoom_factor)
            new_height = int(height * zoom_factor)
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height
            
            zoom_path = output_dir / f"{base_name}_zoom{zoom_factor:.1f}.jpg"
            if not zoom_path.exists():
                zoomed = img.crop((left, top, right, bottom)).resize(original_size)
                zoomed.save(zoom_path, quality=95)
                augmented_paths.append(zoom_path)
            
            print(f"Created {len(augmented_paths)} augmented versions of {image_path.name}")
            return augmented_paths
            
        except Exception as e:
            print(f"Error augmenting {image_path}: {str(e)}")
            return []

    def load_known_faces(self) -> None:
        """
        Load known faces from the faces library directory with caching.
        Each subdirectory name is treated as a person's name, containing multiple images of that person.
        Creates an average face encoding for each person from their multiple images.
        """
        if not self.faces_library_path.exists():
            os.makedirs(self.faces_library_path)
            print(f"Created faces library directory: {self.faces_library_path}")
            return

        # Load existing cache
        cache_data = self._load_cache()
        updated_cache = {}
        
        # Dictionary to store encodings per person
        person_encodings = {}
        
        # Process each subdirectory (person) in the faces library
        for person_dir in self.faces_library_path.iterdir():
            if not person_dir.is_dir():
                continue
                
            person_name = person_dir.name
            person_encodings[person_name] = []
            print(f"\nProcessing faces for: {person_name}")
            
            # Create augmented directory if it doesn't exist
            augmented_dir = person_dir / "augmented"
            if not augmented_dir.exists():
                os.makedirs(augmented_dir)
            
            # Process each original image and create augmentations
            original_images = list(person_dir.glob("*.[jp][pn][g]"))
            for image_path in original_images:
                if "augmented" not in str(image_path):  # Skip augmented images
                    # Create augmented versions
                    augmented_paths = self._augment_face_image(image_path, augmented_dir)
                    
                    # Process original image
                    self._process_face_image(image_path, cache_data, updated_cache, person_encodings, person_name)
                    
                    # Process augmented images
                    for aug_path in augmented_paths:
                        self._process_face_image(aug_path, cache_data, updated_cache, person_encodings, person_name)
            
        # Calculate average encoding for each person
        for person_name, encodings in person_encodings.items():
            if encodings:
                # Convert to numpy array for easier computation
                encodings_array = np.array(encodings)
                # Calculate average encoding
                average_encoding = np.mean(encodings_array, axis=0)
                # Normalize the average encoding
                average_encoding = average_encoding / np.linalg.norm(average_encoding)
                
                self.known_face_encodings.append(average_encoding)
                self.known_face_names.append(person_name)
                print(f"\nCreated average encoding for {person_name} from {len(encodings)} images (including augmentations)")
        
        # Save the updated cache
        self._save_cache(updated_cache)
        
        print(f"\nLoaded {len(self.known_face_encodings)} unique persons with average face encodings")

    def _process_face_image(self, image_path: Path, cache_data: Dict, updated_cache: Dict, 
                          person_encodings: Dict, person_name: str) -> None:
        """Helper function to process a single face image and update relevant data structures."""
        str_path = str(image_path)
        current_metadata = self._get_file_metadata(image_path)
        cached_data = cache_data.get(str_path, {})
        cached_metadata = cached_data.get("metadata", {})
        
        # Check if we need to process this image
        need_processing = True
        if cached_data and cached_metadata:
            if (cached_metadata["modified_time"] == current_metadata["modified_time"] and 
                cached_metadata["size"] == current_metadata["size"]):
                need_processing = False
        
        if need_processing:
            print(f"Processing face: {image_path.name}")
            try:
                image = face_recognition.load_image_file(str_path)
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    encoding = face_encodings[0]
                    updated_cache[str_path] = {
                        "encoding": encoding,
                        "metadata": current_metadata
                    }
                    person_encodings[person_name].append(encoding)
                    print(f"Successfully processed face: {image_path.name}")
                else:
                    print(f"No face found in: {image_path.name}")
                    updated_cache[str_path] = {
                        "encoding": None,
                        "metadata": current_metadata
                    }
            except Exception as e:
                print(f"Error processing {image_path.name}: {str(e)}")
        else:
            print(f"Using cached face encoding for: {image_path.name}")
            if cached_data.get("encoding") is not None:
                encoding = np.array(cached_data["encoding"])
                person_encodings[person_name].append(encoding)
            updated_cache[str_path] = cache_data[str_path]

    def process_video(self, input_path: str, output_path: str, frame_sample_rate: int = 2) -> None:
        """
        Process a video file, detect faces, and save the output with face boxes and labels.
        
        Args:
            input_path (str): Path to input video file
            output_path (str): Path to save output video file
            frame_sample_rate (int): Process every nth frame for face detection
        """
        def log_error(stage: str, error: Exception):
            print(f"\n{'='*50}")
            print(f"ERROR during {stage}:")
            print(f"Error type: {type(error).__name__}")
            print(f"Error message: {str(error)}")
            print("\nFull traceback:")
            traceback.print_exc(file=sys.stdout)
            print('='*50 + '\n')

        def log_timing(stage: str, start_time: float):
            duration = time.time() - start_time
            print(f"[TIMING] {stage}: {duration:.2f} seconds")
            if duration > 5:  # Warning for slow operations
                print(f"⚠️ {stage} took longer than expected!")

        print(f"\n[DEBUG] Starting video processing...")
        total_start = time.time()
        
        video_capture = None
        out = None
        last_progress_update = time.time()
        progress_interval = 5  # Update progress every 5 seconds

        try:
            # Verify input file exists
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input video file not found: {input_path}")

            # Create output directory
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                print(f"[DEBUG] Created/verified output directory: {os.path.dirname(output_path)}")
            except Exception as e:
                log_error("creating output directory", e)
                raise

            # Open video file
            try:
                print("[DEBUG] Attempting to open video file...")
                video_capture = cv2.VideoCapture(input_path)
                if not video_capture.isOpened():
                    raise ValueError(f"Failed to open video file: {input_path}")
            except Exception as e:
                log_error("opening video file", e)
                raise

            # Get video properties
            try:
                frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(video_capture.get(cv2.CAP_PROP_FPS))
                total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                
                print(f"[DEBUG] Video properties:")
                print(f"[DEBUG] - Width: {frame_width}")
                print(f"[DEBUG] - Height: {frame_height}")
                print(f"[DEBUG] - FPS: {fps}")
                print(f"[DEBUG] - Total frames: {total_frames}")
                
                if frame_width == 0 or frame_height == 0 or fps == 0:
                    raise ValueError("Invalid video properties detected")
            except Exception as e:
                log_error("getting video properties", e)
                raise

            # Initialize video writer
            try:
                print("[DEBUG] Initializing video writer...")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                
                if not out.isOpened():
                    raise ValueError("Failed to initialize video writer")
                print("[DEBUG] Video writer initialized successfully")
            except Exception as e:
                log_error("initializing video writer", e)
                raise

            frame_number = 0
            face_locations_prev = []
            face_names_prev = []
            skipped_frames = 0
            processed_frames = 0
            total_detection_time = 0
            total_encoding_time = 0

            print("[DEBUG] Starting frame processing loop...")
            while True:
                try:
                    frame_start = time.time()
                    ret, frame = video_capture.read()
                    if not ret:
                        print("[DEBUG] End of video stream reached")
                        break

                    current_time = time.time()
                    if current_time - last_progress_update >= progress_interval:
                        elapsed_time = current_time - total_start
                        fps = frame_number / elapsed_time if elapsed_time > 0 else 0
                        eta = (total_frames - frame_number) / fps if fps > 0 else 0
                        print(f"\n=== Progress Update ===")
                        print(f"Frame: {frame_number}/{total_frames} ({(frame_number/total_frames*100):.1f}%)")
                        print(f"Average FPS: {fps:.1f}")
                        print(f"ETA: {eta/60:.1f} minutes")
                        print(f"Memory Usage: {psutil.Process().memory_info().rss / (1024*1024):.1f}MB")
                        if processed_frames > 0:
                            print(f"Average detection time: {(total_detection_time/processed_frames):.2f}s")
                            print(f"Average encoding time: {(total_encoding_time/processed_frames):.2f}s")
                        print("====================\n")
                        last_progress_update = current_time

                    if frame_number % frame_sample_rate == 0:
                        processed_frames += 1
                        print(f"\n[DEBUG] Processing frame {frame_number}/{total_frames}")
                        
                        try:
                            # Memory check
                            if frame_number % 100 == 0:  # Check every 100 frames
                                mem = psutil.virtual_memory()
                                if mem.percent > 90:
                                    print(f"⚠️ High memory usage: {mem.percent}%")

                            # Ensure frame is contiguous
                            frame_prep_start = time.time()
                            if not frame.flags['C_CONTIGUOUS']:
                                frame = np.ascontiguousarray(frame)
                            rgb_frame = frame[:, :, ::-1].copy()
                            log_timing("Frame preparation", frame_prep_start)

                            # Face detection
                            detection_start = time.time()
                            print(f"[DEBUG] Starting face detection with {self.detection_model.upper()} model...")
                            try:
                                face_locations = face_recognition.face_locations(rgb_frame, model=self.detection_model, number_of_times_to_upsample=1)
                                detection_time = time.time() - detection_start
                                total_detection_time += detection_time
                                print(f"[DEBUG] Found {len(face_locations)} faces in frame (took {detection_time:.2f}s)")
                            except Exception as e:
                                print("[DEBUG] Face detection failed:")
                                print(f"Frame info: shape={rgb_frame.shape}, dtype={rgb_frame.dtype}")
                                raise

                            # Face encoding
                            encoding_start = time.time()
                            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                            encoding_time = time.time() - encoding_start
                            total_encoding_time += encoding_time
                            print(f"[DEBUG] Generated {len(face_encodings)} face encodings (took {encoding_time:.2f}s)")

                            # Face matching
                            matching_start = time.time()
                            face_names = []
                            for face_encoding in face_encodings:
                                matches = face_recognition.compare_faces(
                                    self.known_face_encodings, 
                                    face_encoding, 
                                    tolerance=0.6
                                )
                                name = "Unknown"
                                if True in matches:
                                    first_match_index = matches.index(True)
                                    name = self.known_face_names[first_match_index]
                                face_names.append(name)
                            log_timing("Face matching", matching_start)

                            face_locations_prev = face_locations
                            face_names_prev = face_names
                        except Exception as e:
                            log_error(f"processing frame {frame_number}", e)
                            raise
                    else:
                        skipped_frames += 1
                        face_locations = face_locations_prev
                        face_names = face_names_prev

                    # Draw results
                    drawing_start = time.time()
                    try:
                        for (top, right, bottom, left), name in zip(face_locations, face_names):
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, name, (left + 6, bottom - 6), 
                                      cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    except Exception as e:
                        log_error(f"drawing results on frame {frame_number}", e)
                        raise
                    log_timing("Drawing results", drawing_start)

                    # Write frame
                    write_start = time.time()
                    try:
                        out.write(frame)
                    except Exception as e:
                        log_error(f"writing frame {frame_number}", e)
                        raise
                    log_timing("Writing frame", write_start)

                    frame_number += 1
                    log_timing("Total frame processing", frame_start)

                except Exception as e:
                    log_error(f"processing frame {frame_number}", e)
                    raise

        except Exception as e:
            log_error("video processing", e)
            raise

        finally:
            print("\n=== Final Statistics ===")
            total_time = time.time() - total_start
            print(f"Total processing time: {total_time/60:.1f} minutes")
            print(f"Frames processed: {processed_frames}")
            print(f"Frames skipped: {skipped_frames}")
            if processed_frames > 0:
                print(f"Average detection time: {(total_detection_time/processed_frames):.2f}s")
                print(f"Average encoding time: {(total_encoding_time/processed_frames):.2f}s")
            print(f"Average FPS: {frame_number/total_time:.1f}")
            print("=====================\n")

            print("[DEBUG] Cleaning up resources...")
            if video_capture is not None:
                video_capture.release()
            if out is not None:
                out.release()
            print("[DEBUG] Cleanup completed")

def main():
    try:
        print("\n[DEBUG] Starting main function...")
        
        # Allow command line argument for model selection
        parser = argparse.ArgumentParser(description='Video Face Recognition')
        parser.add_argument('--model', type=str, choices=['hog', 'cnn'], default='hog',
                          help='Face detection model to use (hog: faster, cnn: more accurate)')
        args = parser.parse_args()
        
        print(f"[DEBUG] Initializing VideoFaceRecognizer with {args.model} model...")
        recognizer = VideoFaceRecognizer('faces_library', detection_model=args.model)
        
        input_video = "tests/protestclip.mp4"
        output_video = "results/protestclip_hog.mp4"
        
        print(f"[DEBUG] Known faces loaded: {len(recognizer.known_face_encodings)}")
        print(f"[DEBUG] Known face names: {recognizer.known_face_names}")
        
        print(f"\nProcessing video: {input_video}")
        print(f"Output will be saved to: {output_video}")
        
        recognizer.process_video(input_video, output_video)
        print("Done! Output saved as", output_video)
        
    except Exception as e:
        print("\n[DEBUG] Fatal error in main:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()