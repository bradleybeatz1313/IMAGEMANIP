from http.server import BaseHTTPRequestHandler
import json


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "status": "ok",
                    "service": "imagemanip",
                    "version": "1.0.0",
                    "endpoints": {
                        "POST /api/process": "Process an image",
                        "GET /api/health": "Health check",
                    },
                }
            ).encode()
        )
