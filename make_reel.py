#!/usr/bin/env python3
"""Builds the reel: hazy intro with big kinetic lyrics -> star-shard reveal -> typed white lyrics.

Timing mirrors the reference reel (audio: reference/dilruba_audio.mp3).
Usage: python3 make_reel.py CLIP1 CLIP2 FONT_TTF OUT.mp4
"""
import random
import subprocess
import sys
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CLIP1, CLIP2, FONT, OUT = sys.argv[1:5]
AUDIO = "reference/dilruba_audio.mp3"
W, H, FPS = 1080, 1920, 30
DUR = 10.8
N = int(DUR * FPS)

# --- timeline -------------------------------------------------------------
T_SWAP0, T_SWAP1 = 3.5, 3.9          # star-shard transition
FADE_OUT = 10.3

STACK = [("TERI", 0.0), ("ADAAON", 0.6), ("KE", 1.0), ("KATRE", 1.2)]
STACK_END = 1.6
SHAB = ("SHABNAMI", 1.6)
FLICKER_FROM = 2.0

# Typed lyrics sit in the top band (face stays in the middle of the main clip).
TYPED = [  # (lines -> [(word, start)]), block end
    ([[("DILRUBA", 4.0)], [("DILNASHEEN", 4.55)]], 6.15),
    ([[("BA-KHUDA", 6.25)], [("TU", 6.95), ("HASEEN", 7.15)]], 8.3),
    ([[("TERE", 8.35), ("JAISI", 8.7)], [("HOOR", 9.1), ("NA", 9.4), ("KOI", 9.7)]], DUR + 1),
]
TYPED_Y, TYPED_LH = 150, 125
LETTER_DT = 0.06


def reader(path, start, dur, vf):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(dur), "-i", path, "-an",
           "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    last = None
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            while True:
                yield last
        last = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        yield last


# CLIP1 = intro (hazy) clip, CLIP2 = main clip. Full frame is kept (no crop, so faces
# are never cut); Snapchat overlays are painted out with delogo instead:
# "Snapchat-1628753313" has a music sticker bottom-right,
# "Snapchat-934401401" has a caption on the left edge.
def clean_for(path):
    if "1628753313" in path:
        return "split[a][b];[b]crop=182:165:356:1065,hflip,boxblur=3[p];[a][p]overlay=538:1065"
    return "split[a][b];[b]crop=90:210:90:805,hflip,boxblur=3[p];[a][p]overlay=0:805"


SLOW2 = 1.25  # main clip is short (6.1s), stretch it to cover 3.5s -> 10.8s
VF1 = f"{clean_for(CLIP1)},scale={W}:{H}:flags=lanczos,fps={FPS},eq=contrast=1.05:saturation=1.1"
VF2 = (f"{clean_for(CLIP2)},setpts={SLOW2}*PTS,scale={W}:{H}:flags=lanczos,fps={FPS},"
       "eq=contrast=1.08:saturation=1.12,colorbalance=rm=0.05:gm=0.0:bm=-0.05")
c1 = reader(CLIP1, 0.4, T_SWAP1 + 0.2, VF1)
c2 = reader(CLIP2, 0.0, 30, VF2)

BIG = ImageFont.truetype(FONT, 215)
SHAB_F = ImageFont.truetype(FONT, 230)
SMALL = ImageFont.truetype(FONT, 118)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def gradient_color(x0, x1, boundary):
    """RGB float array (H,W,3): black left of boundary, orange->red to the right."""
    xs = np.arange(W, dtype=np.float32)
    span = max(x1 - x0, 1)
    t = np.clip((xs - x0) / span, 0, 1)
    orange = np.array([255, 128, 30], np.float32)
    red = np.array([205, 22, 28], np.float32)
    col = orange[None, :] * (1 - t[:, None]) + red[None, :] * t[:, None]
    soft = np.clip((xs - boundary) / 40.0 + 0.5, 0, 1)[:, None]
    row = col * soft  # black (0) where soft==0
    return np.broadcast_to(row[None, :, :], (H, W, 3))


def comp(base, mask, color):
    m = mask[..., None]
    return base * (1 - m) + color * m


def intro_text(t, base):
    """Black condensed lyrics with sweeping orange-red fill (t < T_SWAP1)."""
    mask_img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask_img)
    words = []  # (x0, x1, appear_time)
    if t < STACK_END:
        lh = 205
        y = 330
        for word, ts in STACK:
            if t >= ts:
                wlen = BIG.getlength(word)
                x = (W - wlen) / 2
                d.text((x, y), word, font=BIG, fill=255)
                words.append((x, x + wlen, ts, y, y + lh))
            y += lh
    else:
        word, ts = SHAB
        wlen = SHAB_F.getlength(word)
        x = (W - wlen) / 2
        y = 780
        rng = random.Random(int((t - FLICKER_FROM) / 0.2) if t >= FLICKER_FROM else -1)
        for i, ch in enumerate(word):
            hide = t >= FLICKER_FROM and rng.random() < 0.35
            if not hide:
                d.text((x + SHAB_F.getlength(word[:i]), y), ch, font=SHAB_F, fill=255)
        words.append((x, x + wlen, ts, y, y + 260))
    mask = np.asarray(mask_img, np.float32) / 255.0
    out = base
    for x0, x1, ts, y0, y1 in words:
        k = ease((t - ts - 0.05) / 0.45)
        # colored part grows in from the right edge to ~45% of the word
        boundary = x1 - k * (x1 - x0) * 0.55 + (1 - k) * 60
        if t >= FLICKER_FROM and t < STACK_END + 10 and len(words) == 1:
            boundary += 90 * math.sin(t * 9)  # restless sweep during flicker
        color = gradient_color(x0, x1, boundary)
        band = np.zeros_like(mask)
        band[int(y0) - 20:int(y1) + 60] = mask[int(y0) - 20:int(y1) + 60]
        out = comp(out, band * 0.93, color)
    return out


def typed_text(t, base):
    """White lyrics typed letter-by-letter on the left."""
    block = None
    for lines, end in TYPED:
        if lines[0][0][1] <= t < end:
            block = lines
    if block is None:
        return base
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    y = TYPED_Y
    for line in block:
        full = " ".join(w for w, _ in line)
        shown = []
        for word, ts in line:
            n = int((t - ts) / LETTER_DT) + 1 if t >= ts else 0
            if n > 0:
                shown.append(word[:min(n, len(word))])
        if shown:
            d.text(((W - SMALL.getlength(full)) / 2, y), " ".join(shown), font=SMALL, fill=255)
        y += TYPED_LH
    shadow = np.asarray(img.filter(ImageFilter.GaussianBlur(10)), np.float32) / 255.0
    mask = np.asarray(img, np.float32) / 255.0
    out = comp(base, shadow * 0.75, np.zeros(3, np.float32))
    return comp(out, mask, np.full(3, 248, np.float32))


def haze(frame, amt):
    f = frame.astype(np.float32)
    return f * (1 - amt) + 236.0 * amt


def star_mask(t):
    p = min(max((t - T_SWAP0) / (T_SWAP1 - T_SWAP0), 0.0), 1.0) ** 2.2
    R = 30 + p * 2700
    ang0 = -math.pi / 2 + p * 0.9
    cx, cy = W * 0.52, H * 0.45
    pts = []
    for i in range(10):
        r = R if i % 2 == 0 else R * 0.45
        a = ang0 + i * math.pi / 5
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).polygon(pts, fill=255)
    return np.asarray(img, np.float32) / 255.0, p


enc = subprocess.Popen([
    "ffmpeg", "-v", "error", "-y",
    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
    "-i", AUDIO,
    "-filter_complex", f"[1:a]atrim=0:{DUR},afade=t=out:st={FADE_OUT}:d={DUR - FADE_OUT}[a]",
    "-map", "0:v", "-map", "[a]",
    "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", OUT,
], stdin=subprocess.PIPE)

for i in range(N):
    t = i / FPS
    if t < T_SWAP1:
        a = next(c1)
        hz = 0.62 if t > 0.25 else 0.62 + 0.38 * (1 - t / 0.25)  # open from white
        fa = haze(a, hz)
        fa = intro_text(t, fa)
    if t >= T_SWAP0:
        b = next(c2).astype(np.float32)
        z = 1.0 + 0.06 * (1 - ease((t - T_SWAP0) / 0.6))  # punch-in settle
        if z > 1.001:
            bi = Image.fromarray(b.astype(np.uint8)).resize((int(W * z), int(H * z)), Image.BILINEAR)
            ox, oy = (bi.width - W) // 2, (bi.height - H) // 2
            b = np.asarray(bi.crop((ox, oy, ox + W, oy + H)), np.float32)
        fb = typed_text(t, b)
    if t < T_SWAP0:
        frame = fa
    elif t < T_SWAP1:
        m, p = star_mask(t)
        dark = fa * (1 - 0.55 * p)
        frame = comp(dark, m, fb)
    else:
        frame = fb
    if t > FADE_OUT:
        frame = frame * (1 - (t - FADE_OUT) / (DUR - FADE_OUT))
    enc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())

enc.stdin.close()
enc.wait()
print("done", OUT)
