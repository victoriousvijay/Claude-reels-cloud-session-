#!/usr/bin/env python3
"""Remove the half-offscreen Snapchat caption box from the left edge of a clip.

The box is a semi-transparent dark rounded rectangle (measured: out = 0.575*bg + 11.4)
with static white text. We undo the darkening inside the box, then inpaint only the
text glyphs (found as pixels that stay bright in every frame).
Usage: python3 clean_caption.py IN.mp4 OUT.mp4
"""
import subprocess
import sys

import cv2
import numpy as np

SRC, DST = sys.argv[1:3]
W, H = 720, 1280
X1, Y0, Y1, R = 75, 824, 991, 22      # box: x<X1, Y0<=y<=Y1, corner radius R
GAIN, OFF = 0.575, 11.4

raw = subprocess.run(["ffmpeg", "-v", "error", "-i", SRC, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                     capture_output=True, check=True).stdout
frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)

# soft rounded-rect box mask
box = np.zeros((H, W), np.uint8)
cv2.rectangle(box, (0, Y0), (X1 - R, Y1), 255, -1)
cv2.rectangle(box, (0, Y0 + R), (X1, Y1 - R), 255, -1)
cv2.circle(box, (X1 - R, Y0 + R), R, 255, -1, cv2.LINE_AA)
cv2.circle(box, (X1 - R, Y1 - R), R, 255, -1, cv2.LINE_AA)
boxf = cv2.GaussianBlur(box.astype(np.float32) / 255.0, (5, 5), 1.0)[..., None]

# static text glyphs: bright in every frame
mn = frames[:, Y0 - 10:Y1 + 10, :X1 + 10].mean(3).min(0)
text = np.zeros((H, W), np.uint8)
text[Y0 - 10:Y1 + 10, :X1 + 10] = (mn > 120).astype(np.uint8) * 255
text = cv2.dilate(text, np.ones((7, 7), np.uint8))
# thin ring along the box outline (slight edge mismatch) is inpainted too
k = np.ones((5, 5), np.uint8)
ring = cv2.subtract(cv2.dilate(box, k), cv2.erode(box, k))
ring[:, :2] = 0
text = cv2.bitwise_or(text, ring)

enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-i", SRC, "-map", "0:v", "-map", "1:a?",
                        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "copy", DST],
                       stdin=subprocess.PIPE)
for fr in frames:
    f = fr.astype(np.float32)
    undark = np.clip((f - OFF) / GAIN, 0, 255)
    f = f * (1 - boxf) + undark * boxf
    out = cv2.inpaint(f.astype(np.uint8), text, 6, cv2.INPAINT_TELEA)
    enc.stdin.write(out.tobytes())
enc.stdin.close()
enc.wait()
print("cleaned", DST)
