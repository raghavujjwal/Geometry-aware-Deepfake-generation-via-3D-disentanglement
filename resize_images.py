"""Resize images to max 1024px on the longest side before running DECA."""
import sys
import os
from PIL import Image

MAX_SIZE = 1024
SUPPORTED = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def resize_folder(folder):
    images = [f for f in os.listdir(folder) if f.lower().endswith(SUPPORTED)]
    if not images:
        print(f"No images found in {folder}")
        return
    for fname in images:
        path = os.path.join(folder, fname)
        img = Image.open(path)
        w, h = img.size
        if max(w, h) <= MAX_SIZE:
            print(f"  {fname}: {w}x{h} — already small enough, skipped")
            continue
        scale = MAX_SIZE / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        img.save(path)
        print(f"  {fname}: {w}x{h} -> {new_w}x{new_h}")

if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else '.'
    print(f"Resizing images in: {folder}")
    resize_folder(folder)
    print("Done.")
