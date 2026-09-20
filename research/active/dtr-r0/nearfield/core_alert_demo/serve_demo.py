"""Optional local preview server; double-click launch_demo.cmd needs no Python."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser


class SiteHandler(SimpleHTTPRequestHandler):
    """Restrict file reads to this site's resolved directory."""

    def send_head(self):
        root = Path(self.directory).resolve()
        target = Path(self.translate_path(self.path)).resolve()
        if not target.is_relative_to(root):
            self.send_error(403, "Outside demo folder")
            return None
        return super().send_head()

    def list_directory(self, path):
        self.send_error(404, "No demo page here")
        return None


def make_server(directory: Path, port: int = 0) -> ThreadingHTTPServer:
    directory = directory.resolve()
    if not (directory / "index.html").is_file():
        raise ValueError(f"Missing index.html in demo folder: {directory}")
    handler = partial(SiteHandler, directory=str(directory))
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main():
    parser = argparse.ArgumentParser(description="Preview the offline Core alert demo locally.")
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent,
                        help="Demo site folder (default: beside this script)")
    parser.add_argument("--port", type=int, default=0,
                        help="Local port; 0 chooses a free port (default)")
    parser.add_argument("--browser", action="store_true", help="Open the default browser")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        server = make_server(args.directory, args.port)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    with server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Core alert demo: {url}\nPress Ctrl+C to close this local server.", flush=True)
        if args.browser:
            webbrowser.open(url)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            print("\nLocal demo server stopped.", flush=True)


if __name__ == "__main__":
    main()
