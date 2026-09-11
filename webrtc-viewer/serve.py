"""Loopback static server for the WebRTC viewer, run inside the Isaac container.

Usage: python3 serve.py <port>. Serves this file's directory on 127.0.0.1 only;
the SSH tunnel is the sole way in, exactly like websockify for noVNC. Responses
are never cached so a rewritten connection.json or viewer takes effect on reload.
"""

import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
