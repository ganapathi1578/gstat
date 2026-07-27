import http.server
import socketserver
import os

PORT = 8000

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

if __name__ == '__main__':
    # Serve from the root project directory so we can access /output and /app
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"Serving at http://localhost:{PORT}/app/")
        httpd.serve_forever()
