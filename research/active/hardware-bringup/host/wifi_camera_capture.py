"""Bounded, evidence-preserving local Wi-Fi MJPEG collector (no firmware changes)."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import ipaddress
import json
from pathlib import Path
import re
import time
import urllib.request
from urllib.parse import urlsplit
from atom_capture import ProtocolError, validate_jpeg

MAX_JPEG = 4 * 1024 * 1024
MAX_HEADERS = 8192


def validate_url(url):
    p = urlsplit(url)
    try:
        address = ipaddress.IPv4Address(p.hostname or '')
        valid = any(address in ipaddress.ip_network(n) for n in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'))
        valid = valid and p.port == 81
    except ValueError:
        valid = False
    if not valid or p.scheme != 'http' or p.path != '/stream' or p.query or p.fragment or p.username or p.password:
        raise ValueError('camera URL must be literal private IPv4 http://IP:81/stream')
    return f"http://{address}:81/stream"


class StopCapture(Exception):
    pass


class Body:
    """Retains decoded HTTP body, including malformed and read-ahead bytes."""
    def __init__(self, response, raw, deadline, stop_file=None, clock=time.monotonic):
        self.response, self.raw, self.deadline = response, raw, deadline
        self.stop_file, self.clock = stop_file, clock
        self.pending = bytearray()
        self.digest = hashlib.sha256()
        self.count = 0
        self.stopped = False

    def check(self):
        if self.stop_file and self.stop_file.exists():
            self.stopped = True
            raise StopCapture()
        if self.clock() >= self.deadline:
            raise StopCapture()

    def fill(self):
        self.check()
        data = self.response.read1(16384)
        if not data:
            raise ProtocolError('MJPEG stream ended unexpectedly')
        self.raw.write(data)
        self.raw.flush()
        self.digest.update(data)
        self.count += len(data)
        self.pending.extend(data)

    def line(self):
        while True:
            self.check()
            pos = self.pending.find(b'\n')
            if 0 <= pos < MAX_HEADERS:
                data = bytes(self.pending[:pos+1])
                del self.pending[:pos+1]
                return data
            if len(self.pending) >= MAX_HEADERS:
                raise ProtocolError('multipart header exceeds cap')
            self.fill()

    def exact(self, size):
        while len(self.pending) < size:
            self.fill()
        self.check()
        data = bytes(self.pending[:size])
        del self.pending[:size]
        return data


def receive(body, boundary):
    line = body.line()
    if line == b'\r\n':
        line = body.line()
    if line.rstrip(b'\r\n') != b'--'+boundary:
        raise ProtocolError('unexpected multipart boundary')
    headers, total = {}, 0
    while True:
        line = body.line()
        total += len(line)
        if total > MAX_HEADERS:
            raise ProtocolError('multipart headers exceed cap')
        if line == b'\r\n':
            break
        try:
            key, value = line.decode('ascii').strip().split(':', 1)
        except ValueError as exc:
            raise ProtocolError('malformed multipart header') from exc
        key = key.lower()
        if key in headers:
            raise ProtocolError('duplicate multipart header')
        headers[key] = value.strip()
    if headers.get('content-type') != 'image/jpeg':
        raise ProtocolError('multipart payload is not image/jpeg')
    try:
        size = int(headers['content-length'])
        seq = int(headers['x-frame-sequence'])
        timestamps = {name: int(headers[source]) for name, source in (
            ('capture_timestamp_us', 'x-capture-timestamp-us'),
            ('jpeg_ready_timestamp_us', 'x-jpeg-ready-timestamp-us'),
            ('device_send_start_timestamp_us', 'x-device-send-start-timestamp-us'))}
        boot = headers['x-sequence-id']
    except (KeyError, ValueError) as exc:
        raise ProtocolError('missing/invalid camera metadata') from exc
    if not 4 <= size <= MAX_JPEG or not 0 <= seq < 2**32 or not boot or any(v < 0 or v >= 2**64 for v in timestamps.values()):
        raise ProtocolError('camera metadata outside bounds')
    jpeg = body.exact(size)
    return {'type': 'jpeg', 'seq': seq, 'sequence_id': boot, 'bytes': size,
            'transport': 'wifi_mjpeg', **timestamps}, jpeg


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProtocolError('camera HTTP redirects are disabled')


def acquire(url, output, seconds, stop_file=None, *, opener=None, clock=time.monotonic):
    url = validate_url(url)
    if not 0 < seconds <= 900:
        raise ValueError('seconds must be in (0,900]')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    stop_file = Path(stop_file) if stop_file else None
    started = clock()
    manifest = {'schema': 'hardware-bringup.wifi-jpeg.v1', 'url': url,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'requested_seconds': seconds,
        'raw_format': 'HTTP transfer-decoded multipart body; excludes HTTP chunk framing',
        'clock_boundary': 'device timestamps retained; host receipt is not exposure synchronization',
        'io_timeout_seconds': 3}
    frames = failures = 0
    error = None
    body = None
    stopped = False
    response = None
    with (output/'raw.bin').open('xb') as raw, (output/'frames.jsonl').open('x', encoding='utf-8') as records, (output/'events.jsonl').open('x', encoding='utf-8') as events:
        try:
            if stop_file and stop_file.exists():
                stopped = True
            else:
                if opener is None:
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open
                response = opener(url, timeout=3)
                body = Body(response, raw, started+seconds, stop_file, clock)
                content_type = response.headers.get('Content-Type', '')
                match = re.search(r'boundary="?([A-Za-z0-9_-]{1,70})"?(?:;|$)', content_type)
                if not content_type.lower().startswith('multipart/x-mixed-replace') or not match:
                    raise ProtocolError('invalid multipart Content-Type/boundary')
                boundary = match.group(1).encode('ascii')
                while True:
                    header, jpeg = receive(body, boundary)
                    receipt = time.monotonic_ns()
                    filename = f'frame-{frames:06d}-seq-{header["seq"]:010d}.jpg'
                    (output/filename).write_bytes(jpeg)
                    record = {'header': header, 'filename': filename, 'host_received_monotonic_ns': receipt,
                              'sha256': hashlib.sha256(jpeg).hexdigest(), 'jpeg_validated': False}
                    try:
                        from PIL import Image
                        with Image.open(io.BytesIO(jpeg)) as image:
                            header['width'], header['height'] = image.size
                        validate_jpeg(jpeg, header)
                        record['jpeg_validated'] = True
                    finally:
                        records.write(json.dumps(record)+'\n')
                        records.flush()
                    frames += 1
        except StopCapture:
            stopped = body.stopped if body else stopped
        except KeyboardInterrupt:
            stopped = True
        except Exception as exc:
            failures += 1
            error = f'{type(exc).__name__}: {exc}'
            events.write(json.dumps({'kind': 'acquisition_error', 'error': error, 'host_received_monotonic_ns': time.monotonic_ns()})+'\n')
        finally:
            if response is not None:
                response.close()
            result = {'frames': frames, 'failure_count': failures, 'stopped_by_request': stopped,
                      'acquisition_error': error, 'elapsed_seconds': clock()-started,
                      'raw_bytes': body.count if body else 0,
                      'raw_sha256': body.digest.hexdigest() if body else hashlib.sha256(b'').hexdigest()}
            manifest['result'] = result
            (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
            (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    result = acquire(args.url, args.output, args.seconds, args.stop_file)
    print(json.dumps(result))
    return int(bool(result['acquisition_error']))


if __name__ == '__main__':
    raise SystemExit(main())
