import os
from pathlib import Path
import json
from PIL import Image, ImageEnhance
import shutil

class FaceAugmenter:
    def __init__(self, faces_library_path: str):
        self.faces_library_path = Path(faces_library_path)
        self.augmentation_record = self.faces_library_path / ".augmentation_record.json"
        self.processed_images = self._load_record()

    def _load_record(self) -> dict:
        """Load record of which images have been augmented."""
        if self.augmentation_record.exists():
            try:
                with open(self.augmentation_record, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Record file corrupted, will rebuild")
        return {}

    def _save_record(self):
        """Save record of processed images."""
        with open(self.augmentation_record, 'w') as f:
            json.dump(self.processed_images, f, indent=2)

    def _get_augmented_name(self, base_name: str, transform_type: str, param: str = "") -> str:
        """Generate consistent naming for augmented images."""
        return f"{base_name}__aug_{transform_type}{param}.jpg"

    def augment_image(self, image_path: Path) -> list:
        """Create augmented versions of a face image with various transformations."""
        augmented_paths = []
        try:
            # Open image with PIL
            img = Image.open(image_path)
            base_name = image_path.stem
            original_size = img.size
            parent_dir = image_path.parent

            # 1. Horizontal Flip
            flip_name = self._get_augmented_name(base_name, "flip")
            flip_path = parent_dir / flip_name
            if not flip_path.exists():
                img.transpose(Image.FLIP_LEFT_RIGHT).save(flip_path, quality=95)
                augmented_paths.append(flip_path)

            # 2. Rotations
            for angle in [-10, -5, 5, 10]:
                rot_name = self._get_augmented_name(base_name, "rot", str(angle))
                rot_path = parent_dir / rot_name
                if not rot_path.exists():
                    rotated = img.rotate(angle, expand=True)
                    rotated = rotated.resize(original_size)
                    rotated.save(rot_path, quality=95)
                    augmented_paths.append(rot_path)

            # 3. Brightness
            for factor in [0.8, 1.2]:
                bright_name = self._get_augmented_name(base_name, "bright", f"{factor:.1f}")
                bright_path = parent_dir / bright_name
                if not bright_path.exists():
                    enhancer = ImageEnhance.Brightness(img)
                    enhancer.enhance(factor).save(bright_path, quality=95)
                    augmented_paths.append(bright_path)

            # 4. Contrast
            for factor in [0.8, 1.2]:
                contrast_name = self._get_augmented_name(base_name, "contrast", f"{factor:.1f}")
                contrast_path = parent_dir / contrast_name
                if not contrast_path.exists():
                    enhancer = ImageEnhance.Contrast(img)
                    enhancer.enhance(factor).save(contrast_path, quality=95)
                    augmented_paths.append(contrast_path)

            # 5. Zoom
            zoom_factor = 0.9
            zoom_name = self._get_augmented_name(base_name, "zoom", f"{zoom_factor:.1f}")
            zoom_path = parent_dir / zoom_name
            if not zoom_path.exists():
                width, height = img.size
                new_width = int(width * zoom_factor)
                new_height = int(height * zoom_factor)
                left = (width - new_width) // 2
                top = (height - new_height) // 2
                right = left + new_width
                bottom = top + new_height
                zoomed = img.crop((left, top, right, bottom)).resize(original_size)
                zoomed.save(zoom_path, quality=95)
                augmented_paths.append(zoom_path)

            if augmented_paths:
                print(f"Created {len(augmented_paths)} augmented versions of {image_path.name}")
            return augmented_paths

        except Exception as e:
            print(f"Error augmenting {image_path}: {str(e)}")
            return []

    def process_directory(self, person_dir: Path):
        """Process all original images in a person's directory."""
        if not person_dir.is_dir():
            return

        # Get list of original images (not augmented ones)
        original_images = [f for f in person_dir.glob("*.[jp][pn][g]") 
                         if "__aug_" not in f.name]

        for image_path in original_images:
            str_path = str(image_path)
            
            # Skip if already processed
            if str_path in self.processed_images:
                print(f"Skipping {image_path.name} - already augmented")
                continue

            print(f"\nProcessing: {image_path.name}")
            augmented_paths = self.augment_image(image_path)
            
            if augmented_paths:
                self.processed_images[str_path] = {
                    "augmented_versions": [str(p) for p in augmented_paths],
                    "timestamp": str(datetime.now())
                }
                self._save_record()

    def process_library(self):
        """Process all person directories in the face library."""
        if not self.faces_library_path.exists():
            print(f"Face library not found: {self.faces_library_path}")
            return

        print(f"Processing face library: {self.faces_library_path}")
        
        # Process each person's directory
        for person_dir in self.faces_library_path.iterdir():
            if person_dir.is_dir():
                print(f"\nProcessing person: {person_dir.name}")
                self.process_directory(person_dir)

        print("\nAugmentation complete!")
        print(f"Processed images record saved to: {self.augmentation_record}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Augment face images with variations')
    parser.add_argument('--library', type=str, default='faces_library',
                      help='Path to faces library directory')
    args = parser.parse_args()

    augmenter = FaceAugmenter(args.library)
    augmenter.process_library()

if __name__ == "__main__":
    from datetime import datetime
    main() 