from mss import mss
from PIL import Image
from model import ViT
import torch
import time
import numpy as np
from colorama import Fore
import cv2
import albumentations as A
import os
import socket
import sys

# ----- single-instance lock (simple) -----
PID = os.getpid()
print(f"Starting candlestick_prediction.py, PID={PID}")
LOCK_PORT = 49999  # change if already in use
_lock_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    _lock_sock.bind(("127.0.0.1", LOCK_PORT))
except OSError:
    print("Another instance appears to be running (port lock failed). Exiting.")
    sys.exit(1)

# ----- model (unchanged) -----
model = ViT()
model.load_state_dict(torch.load('checkpoints/25_model.pt', map_location=torch.device('cpu')))
model.eval()

classes = {
    '0': 'doji',
    '1': 'bullish_engulfing',
    '2': 'bearish_engulfing',
    '3': 'morning_star',
    '4': 'evening_star',
}
bar_colors = [
    (156, 220, 235), (166, 207, 140), (236, 171, 193), (202, 163, 232), (255, 128, 128)
]

WINDOW_NAME = "CandlestickPrediction"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
# initial size, final position set later once capture width is known
cv2.resizeWindow(WINDOW_NAME, 1200, 800)

softy = torch.nn.Softmax(dim=1)

# --- capture defaults & lock ---
CAPTURE_LEFT = 0
CAPTURE_TOP = 0
CAPTURE_WIDTH_DEFAULT = 1349  # keep overlay-start (1350) outside capture

print_every = 10  # only print to console every N frames

try:
    with mss() as sct:
        monitor = sct.monitors[0]
        MON_W = monitor['width']
        MON_H = monitor['height']

        # Ensure capture width doesn't exceed monitor width
        CAPTURE_WIDTH = min(CAPTURE_WIDTH_DEFAULT, MON_W)
        CAPTURE_HEIGHT = MON_H

        print(f"Monitor resolution detected: {MON_W}x{MON_H}. Capturing left area {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}.")

        # Move the window to the right of the captured area so it is not in screenshots.
        # If CAPTURE_WIDTH is in pixels from left, put the window's left edge at CAPTURE_WIDTH + 20
        overlay_pos_x = CAPTURE_WIDTH + 20
        overlay_pos_y = 30
        try:
            cv2.moveWindow(WINDOW_NAME, overlay_pos_x, overlay_pos_y)
        except Exception:
            # moveWindow may fail in some environments; ignore if it does
            pass

        # Build transforms relative to the capture size (safe)
        init_x_min = 0
        init_y_min = 170
        init_x_max = min(1920, CAPTURE_WIDTH)
        init_y_max = min(1080, CAPTURE_HEIGHT)
        if init_y_min >= init_y_max:
            init_y_min = 0
            init_y_max = CAPTURE_HEIGHT
        if init_x_min >= init_x_max:
            init_x_min = 0
            init_x_max = CAPTURE_WIDTH

        transforms = A.Compose(
            [
                A.Crop(x_min=init_x_min, y_min=init_y_min, x_max=init_x_max, y_max=init_y_max),
                A.Resize(700, 500),
                A.Resize(224, 224),
                A.Crop(x_min=130, y_min=43, x_max=202, y_max=147),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensorV2(),
            ]
        )

        frame_idx = 0
        while True:
            frame_idx += 1

            bbox = {"left": CAPTURE_LEFT, "top": CAPTURE_TOP, "width": CAPTURE_WIDTH, "height": CAPTURE_HEIGHT}
            sct_image = sct.grab(bbox)
            raw_image = Image.frombytes("RGB", sct_image.size, sct_image.rgb)

            cap_np = np.array(raw_image)  # RGB numpy

            # safe transforms (already clamped)
            try:
                res = transforms(image=cap_np)
                img = res['image']
            except Exception as e:
                # fallback: resize to 224x224 and convert to tensor (best effort)
                print("Transforms failed, using fallback resize:", e)
                tmp = cv2.resize(cv2.cvtColor(cap_np, cv2.COLOR_RGB2BGR), (224, 224))
                tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(tmp.transpose(2, 0, 1)).float() / 255.0

            # inference
            with torch.no_grad():
                preds = model(torch.unsqueeze(img, dim=0))
                probs = softy(preds)
                prediction = int(torch.argmax(probs, dim=-1)[0].item())
                probability = float(probs[0][prediction].item())

            # light printing
            if frame_idx % print_every == 0:
                print(Fore.LIGHTYELLOW_EX + f"Frame {frame_idx}: pred={prediction} prob={probability:.4f}" + Fore.RESET)

            # prepare render image (BGR)
            render_image = cv2.cvtColor(cap_np, cv2.COLOR_RGB2BGR)
            h, w = render_image.shape[:2]

            # overlay block (relative)
            overlay = render_image.copy()
            block_x1 = int(w * 0.55)
            block_x2 = int(w * 0.98)
            block_y1 = int(h * 0.05)
            block_y2 = int(h * 0.45)
            cv2.rectangle(overlay, (block_x1, block_y1), (block_x2, block_y2), (128, 128, 128), -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, render_image, 1.0 - alpha, 0, dst=render_image)

            # bars & labels
            base_x = int(w * 0.56)
            bar_max_width = int(w * 0.35)
            text_x = base_x + 10
            for x in range(len(classes.keys())):
                class_prob = float(probs[0][x].item())
                label = f"{classes[str(x)]} - {round(class_prob, 3)}"
                y_min = (x + 1) * int(h * 0.06) + int(h * 0.08)
                y_max = y_min + int(h * 0.045)
                x_max = base_x + int(bar_max_width * class_prob)
                cv2.rectangle(render_image, (base_x, y_min), (x_max, y_max), bar_colors[x], -1)
                cv2.putText(render_image, label, (text_x, y_min + int(h * 0.035)),
                            cv2.FONT_HERSHEY_DUPLEX, max(0.6, h/1200), (0, 0, 0), 2, cv2.LINE_AA)

            # predicted label box
            box_x1 = int(w * 0.73)
            box_x2 = int(w * 0.96)
            box_y1 = int(h * 0.46)
            box_y2 = int(h * 0.6)
            cv2.rectangle(render_image, (box_x1, box_y1), (box_x2, box_y2), bar_colors[int(prediction)], 6)
            label_bg_x1 = int(w * 0.55)
            label_bg_x2 = int(w * 0.96)
            label_bg_y1 = int(h * 0.02)
            label_bg_y2 = int(h * 0.08)
            cv2.rectangle(render_image, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), bar_colors[int(prediction)], -1)
            label_text = f"{classes[str(int(prediction))]} - {round(probability, 3)}"
            cv2.putText(render_image, label_text, (int(w * 0.57), int(h * 0.055)),
                        cv2.FONT_HERSHEY_DUPLEX, max(0.9, h/600), (0, 0, 0), 3, cv2.LINE_AA)

            # show window (reused)
            cv2.imshow(WINDOW_NAME, render_image)

            # ensure overlay window remains outside capture area in case something moved it
            try:
                cv2.moveWindow(WINDOW_NAME, overlay_pos_x, overlay_pos_y)
            except Exception:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # small sleep for CPU
            time.sleep(0.03)

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    try:
        _lock_sock.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
