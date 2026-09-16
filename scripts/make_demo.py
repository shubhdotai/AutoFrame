"""Create a muted, side-by-side comparison. Requires opencv-python and numpy.

Usage: python scripts/make_demo.py original.mp4 reframed.mp4 --output demo.mp4
Both videos should start at the same moment. Stops when either video ends.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def place_frame(canvas, frame, x, width):
    """Fit a frame below its title, keeping its original aspect ratio."""
    top, padding = 64, 16
    height = canvas.shape[0] - top - padding
    available_width = width - 2 * padding
    scale = min(available_width / frame.shape[1], height / frame.shape[0])
    w = max(1, round(frame.shape[1] * scale))
    h = max(1, round(frame.shape[0] * scale))
    resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    left = x + (width - w) // 2
    y = top + (height - h) // 2
    canvas[y:y + h, left:left + w] = resized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("reframed", type=Path)
    parser.add_argument("--output", type=Path, default=Path("demo.mp4"))
    args = parser.parse_args()
    for path in (args.original, args.reframed):
        if path.suffix.lower() != ".mp4" or not path.is_file():
            parser.error(f"Input must be an existing MP4 file: {path}")
    if args.output.resolve() in {args.original.resolve(), args.reframed.resolve()}:
        parser.error("Output must be different from both input videos.")
    if args.output.exists():
        parser.error("Output already exists; choose a different filename.")
    if args.output.suffix.lower() != ".mp4":
        parser.error("Output must have an .mp4 extension.")

    captures = []
    writer = None
    try:
        rates = []
        for path in (args.original, args.reframed):
            capture = cv2.VideoCapture(str(path))
            captures.append(capture)
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"Cannot read video or frame rate: {path}")
            rates.append(fps)

        # Fixed canvas: a wide panel for the original and a portrait panel.
        background = np.full((720, 1280, 3), 24, dtype=np.uint8)
        for title, x, width in (("Original video", 0, 880), ("Reframed", 880, 400)):
            text_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
            cv2.putText(background, title, (x + (width - text_width) // 2, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                 rates[0], (1280, 720))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output: {args.output}")

        indices = [-1, -1]
        frames = [None, None]
        count = 0
        while True:
            # Match elapsed time even when the videos have different frame rates.
            for i, capture in enumerate(captures):
                target = math.floor(count * rates[i] / rates[0] + 1e-9)
                while indices[i] < target:
                    ok, frames[i] = capture.read()
                    indices[i] += 1
                    if not ok:
                        break
                if frames[i] is None:
                    break
            if any(frame is None for frame in frames):
                break
            canvas = background.copy()
            place_frame(canvas, frames[0], 0, 880)
            place_frame(canvas, frames[1], 880, 400)
            writer.write(canvas)
            count += 1

        if count == 0:
            raise ValueError("One of the videos contains no readable frames.")
        print(f"Saved {args.output}: {count / rates[0]:.2f}s, 1280x720, muted")
    finally:
        for capture in captures:
            capture.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
