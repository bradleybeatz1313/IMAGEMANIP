from http.server import BaseHTTPRequestHandler
import json
import base64
import io
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image, ImageEnhance
import random

MAX_DIM = 2048


class FrequencyDomainProcessor:
    def __init__(self, strength=0.12):
        self.strength = strength

    def modify_frequency_spectrum(self, arr):
        result = np.zeros_like(arr, dtype=np.float32)
        for ch in range(arr.shape[2]):
            f = fftshift(fft2(arr[:, :, ch].astype(np.float32)))
            rows, cols = f.shape
            cr, cc = rows // 2, cols // 2
            y, x = np.ogrid[:rows, :cols]
            dist = np.sqrt((x - cc) ** 2 + (y - cr) ** 2)
            mask = (dist > rows * 0.08) & (dist < rows * 0.45)
            mod = np.random.normal(1.0, self.strength, f.shape)
            f[mask] *= mod[mask]
            result[:, :, ch] = np.real(ifft2(ifftshift(f)))
        return np.clip(result, 0, 255).astype(np.uint8)

    def add_lens_characteristics(self, arr):
        arr = arr.astype(np.float32)
        rows, cols = arr.shape[:2]
        cr, cc = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - cc) ** 2 + (y - cr) ** 2)
        max_dist = np.sqrt(cc ** 2 + cr ** 2)
        vignette = 1.0 - 0.12 * (dist / max_dist) ** 2.2
        scales = [1.002, 1.0, 0.998]
        for ch in range(arr.shape[2]):
            arr[:, :, ch] *= vignette * scales[ch]
        return np.clip(arr, 0, 255).astype(np.uint8)


class OrganicNoiseInjector:
    PROFILES = {
        "canon": {"base": 0.012, "color": 0.008, "pattern": 0.004},
        "sony": {"base": 0.010, "color": 0.006, "pattern": 0.003},
        "nikon": {"base": 0.014, "color": 0.009, "pattern": 0.005},
        "iphone": {"base": 0.008, "color": 0.005, "pattern": 0.002},
    }

    def apply(self, arr, camera="canon"):
        p = self.PROFILES.get(camera, self.PROFILES["canon"])
        h, w, c = arr.shape
        result = arr.astype(np.float32)
        for ch in range(c):
            ns = p["base"] * (0.8 + 0.4 * random.random())
            result[:, :, ch] += np.random.normal(0, ns * 255, (h, w))
        result += np.random.normal(0, p["color"] * 255, (h, w, c))
        n_defects = int(h * w * 0.0002)
        for _ in range(n_defects):
            px, py = random.randint(0, w - 1), random.randint(0, h - 1)
            if random.random() < 0.7:
                result[py, px] = [255, random.randint(200, 255), random.randint(150, 255)]
            else:
                result[py, px] = [0, random.randint(0, 50), random.randint(0, 30)]
        s = p["pattern"]
        bayer = np.ones((h, w))
        bayer[0::2, 0::2] *= 1.0 + s
        bayer[0::2, 1::2] = 1.0 - s * 0.5
        bayer[1::2, 0::2] = 1.0 - s * 0.5
        bayer[1::2, 1::2] = 1.0 + s * 0.7
        channel_mods = [1.0 + s * 0.3, 1.0, 1.0 - s * 0.2]
        for ch in range(c):
            result[:, :, ch] *= bayer * channel_mods[ch]
        return np.clip(result, 0, 255).astype(np.uint8)


class CompressionProcessor:
    def cycle(self, image, cycles=3):
        current = image.copy()
        for i in range(cycles):
            buf = io.BytesIO()
            if i % 2 == 0:
                q = random.choice([94, 87, 91, 85, 89, 92])
                current.save(
                    buf,
                    format="JPEG",
                    quality=q,
                    optimize=True,
                    subsampling=random.choice(["4:4:4", "4:2:2", "4:2:0"]),
                )
            else:
                current.save(
                    buf, format="PNG", compress_level=random.choice([6, 7, 8]), optimize=True
                )
            buf.seek(0)
            current = Image.open(buf).copy()
            if i < cycles - 1:
                a = np.array(current).astype(np.float32)
                for ch in range(a.shape[2]):
                    a[:, :, ch] += np.random.normal(0, 0.003 * 255, a[:, :, ch].shape)
                current = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        return current


class AdversarialProcessor:
    def __init__(self, epsilon=8 / 255):
        self.epsilon = epsilon

    def apply(self, arr):
        h, w, c = arr.shape
        perturbation = np.zeros((h, w, c), dtype=np.float32)
        cb = np.zeros((h, w))
        cb[0::2, 1::2] = 1
        cb[1::2, 0::2] = 1
        cb = (cb - 0.5) * 2
        xv = np.linspace(0, 8 * np.pi, w)
        yv = np.linspace(0, 8 * np.pi, h)
        X, Y = np.meshgrid(xv, yv)
        s1 = np.sin(X) * np.cos(Y)
        s2 = np.sin(1.7 * X + 0.3 * Y) * np.cos(0.7 * X + 1.1 * Y)
        weights = [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]]
        for ch in range(c):
            cw = 0.8 + 0.4 * random.random()
            pattern = weights[ch][0] * cb + weights[ch][1] * s1 + weights[ch][2] * s2
            perturbation[:, :, ch] = pattern * self.epsilon * 255 * cw
        return np.clip(arr.astype(np.float32) + perturbation, 0, 255).astype(np.uint8)


class EntropyManager:
    def manage(self, arr):
        entropy = self._entropy(arr.flatten())
        if entropy > 7.6:
            return self._reduce(arr)
        elif entropy < 6.8:
            return self._increase(arr)
        return arr

    def _entropy(self, data):
        if len(data) == 0:
            return 0
        counts = np.bincount(data.astype(int))
        probs = counts / len(data)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _reduce(self, arr):
        r = arr.astype(np.float32)
        h, w = arr.shape[:2]
        X, Y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        grad = ((X + Y) / 2 - 0.5) * 10
        for ch in range(arr.shape[2]):
            r[:, :, ch] += grad
        return np.clip(r, 0, 255).astype(np.uint8)

    def _increase(self, arr):
        r = arr.astype(np.float32) + np.random.normal(0, 2, arr.shape)
        return np.clip(r, 0, 255).astype(np.uint8)


class ImageProcessor:
    def __init__(self):
        self.freq = FrequencyDomainProcessor()
        self.noise = OrganicNoiseInjector()
        self.compress = CompressionProcessor()
        self.adversarial = AdversarialProcessor()
        self.entropy = EntropyManager()

    def process(self, image, intensity="high", camera="canon"):
        arr = np.array(image)
        arr = self.freq.modify_frequency_spectrum(arr)
        arr = self.freq.add_lens_characteristics(arr)
        arr = self.adversarial.apply(arr)
        arr = self.noise.apply(arr, camera)
        arr = self.entropy.manage(arr)
        img = Image.fromarray(arr)
        cycles = {"low": 1, "medium": 2, "high": 3}.get(intensity, 2)
        img = self.compress.cycle(img, cycles)
        if intensity == "high":
            img = ImageEnhance.Contrast(img).enhance(1.02)
            img = ImageEnhance.Sharpness(img).enhance(0.98)
        return img


def _resize(image):
    w, h = image.size
    if w > MAX_DIM or h > MAX_DIM:
        ratio = min(MAX_DIM / w, MAX_DIM / h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return image


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            img_data = base64.b64decode(body["image"])
            image = Image.open(io.BytesIO(img_data))
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = _resize(image)

            result = ImageProcessor().process(
                image, body.get("intensity", "high"), body.get("camera", "canon")
            )

            buf = io.BytesIO()
            result.save(buf, format="JPEG", quality=95, optimize=True)
            out_b64 = base64.b64encode(buf.getvalue()).decode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"image": out_b64}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
