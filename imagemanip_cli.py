#!/usr/bin/env python3
"""
IMAGEMANIP CLI - Process images via the IMAGEMANIP API.

Usage:
    python imagemanip_cli.py photo.jpg
    python imagemanip_cli.py photo.png --camera iphone --intensity medium
    python imagemanip_cli.py https://example.com/image.jpg
    python imagemanip_cli.py photo.jpg -o processed.jpg
"""

import argparse
import base64
import json
import os
import sys
import urllib.request
import urllib.error

API_URL = "https://imagemanip.vercel.app/api/process"


def fetch_image_bytes(source):
    if source.startswith(("http://", "https://")):
        print(f"Downloading {source}")
        req = urllib.request.Request(source, headers={"User-Agent": "IMAGEMANIP-CLI/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    path = os.path.expanduser(source)
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "rb") as f:
        return f.read()


def process(image_bytes, camera="canon", intensity="high"):
    payload = json.dumps({
        "image": base64.b64encode(image_bytes).decode(),
        "camera": camera,
        "intensity": intensity,
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print("Processing... ", end="", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            msg = json.loads(body).get("error", body)
        except Exception:
            msg = body
        print("failed")
        print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print("failed")
        print(f"Error: {e.reason}", file=sys.stderr)
        sys.exit(1)

    print("done")
    return base64.b64decode(data["image"])


def default_output(source):
    if source.startswith(("http://", "https://")):
        name = source.rstrip("/").split("/")[-1].split("?")[0]
        base, _ = os.path.splitext(name) if "." in name else (name, "")
        return f"{base or 'image'}_processed.jpg"
    base, _ = os.path.splitext(os.path.basename(source))
    return f"{base}_processed.jpg"


def main():
    parser = argparse.ArgumentParser(
        description="Process images via the IMAGEMANIP API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python imagemanip_cli.py photo.jpg\n"
               "  python imagemanip_cli.py photo.png -c iphone -i medium\n"
               "  python imagemanip_cli.py https://example.com/img.jpg -o out.jpg",
    )
    parser.add_argument("source", help="Image file path or URL")
    parser.add_argument("-o", "--output", help="Output file path (default: <name>_processed.jpg)")
    parser.add_argument(
        "-c", "--camera",
        choices=["canon", "sony", "nikon", "iphone"],
        default="canon",
        help="Camera profile (default: canon)",
    )
    parser.add_argument(
        "-i", "--intensity",
        choices=["low", "medium", "high"],
        default="high",
        help="Processing intensity (default: high)",
    )
    args = parser.parse_args()

    image_bytes = fetch_image_bytes(args.source)
    size_mb = len(image_bytes) / (1024 * 1024)
    print(f"Image loaded ({size_mb:.1f} MB)")

    result = process(image_bytes, args.camera, args.intensity)

    output = args.output or default_output(args.source)
    with open(output, "wb") as f:
        f.write(result)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
