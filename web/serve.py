#!/usr/bin/env python3
# serve.py - tiny no-cache static dev server. Finds a free port in 8080-8099
# (so it won't clash with another project's server). Open the URL it prints.
import http.server, socketserver, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

httpd = None
for port in range(8080, 8100):
    try:
        httpd = socketserver.TCPServer(("", port), Handler)
        break
    except OSError:
        continue

if httpd is None:
    raise SystemExit("No free port in 8080-8099.")

print(f"serving {os.getcwd()} at http://localhost:{port}")
httpd.serve_forever()
