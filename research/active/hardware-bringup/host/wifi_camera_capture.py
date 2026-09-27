"""Bounded, evidence-preserving local Wi-Fi MJPEG collector (no firmware changes)."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import urllib.error
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


class TransportError(Exception):
    """Only connection/open/read failures may trigger reconnection."""



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
        try:
            data = self.response.read1(16384)
        except http.client.IncompleteRead as exc:
            if exc.partial:
                self.retain(exc.partial)
            raise TransportError(f'{type(exc).__name__}: {exc}') from exc
        except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
            raise TransportError(f'{type(exc).__name__}: {exc}') from exc
        if not data:
            raise TransportError('MJPEG stream ended unexpectedly')
        self.retain(data)

    def retain(self, data):
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


class RawEvidence:
    """One digest over all decoded body bytes from every connection."""
    def __init__(self, stream):
        self.stream, self.digest, self.count = stream, hashlib.sha256(), 0

    def write(self, data):
        self.stream.write(data)
        self.digest.update(data)
        self.count += len(data)

    def flush(self):
        self.stream.flush()


def acquire(url, output, seconds, stop_file=None, *, opener=None, clock=time.monotonic,
            sleep=time.sleep, stamp=time.monotonic_ns):
    url = validate_url(url)
    if not 0 < seconds <= 900:
        raise ValueError('seconds must be in (0,900]')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    stop_file = Path(stop_file) if stop_file else None
    started = clock()
    deadline = started + seconds
    manifest = {'schema': 'hardware-bringup.wifi-jpeg.v1', 'url': url,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'requested_seconds': seconds,
        'raw_format': 'concatenated HTTP transfer-decoded multipart bodies; excludes HTTP chunk framing; events include raw offsets at connections',
        'clock_boundary': 'device timestamps retained; host receipt is not exposure synchronization',
        'io_timeout_seconds': 3, 'retry_delays_seconds': [1, 2, 4, 5],
        'retry_policy': 'network/EOF only; original deadline unchanged; malformed protocol/JPEG terminates'}
    frames = failures = reconnects = 0
    error = last_error = None
    stopped = False
    response = None
    if opener is None:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open

    def check():
        if stop_file and stop_file.exists():
            raise StopCapture()
        if clock() >= deadline:
            raise StopCapture()

    def status(state, retry_at=None):
        obj = {'state': state, 'reconnects': reconnects, 'last_error': last_error}
        if retry_at is not None:
            obj['next_retry_monotonic_ns'] = stamp() + int(max(0, retry_at-clock())*1e9)
        tmp = output/'status.json.tmp'
        tmp.write_text(json.dumps(obj)+'\n', encoding='utf-8')
        tmp.replace(output/'status.json')

    with (output/'raw.bin').open('xb') as raw_file, (output/'frames.jsonl').open('x', encoding='utf-8') as records, (output/'events.jsonl').open('x', encoding='utf-8') as events:
        raw = RawEvidence(raw_file)

        def event(kind, **fields):
            events.write(json.dumps({'kind': kind, 'host_received_monotonic_ns': stamp(),
                'attempt': reconnects, 'raw_offset': raw.count, **fields})+'\n')
            events.flush()

        try:
            status('connecting')
            while True:
                check()
                try:
                    try:
                        response = opener(url, timeout=min(3, max(.001, deadline-clock())))
                    except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
                        if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                            raise
                        raise TransportError(f'{type(exc).__name__}: {exc}') from exc
                    check()
                    body = Body(response, raw, deadline, stop_file, clock)
                    content_type = response.headers.get('Content-Type', '')
                    match = re.search(r'boundary="?([A-Za-z0-9_-]{1,70})"?(?:;|$)', content_type)
                    if not content_type.lower().startswith('multipart/x-mixed-replace') or not match:
                        raise ProtocolError('invalid multipart Content-Type/boundary')
                    boundary = match.group(1).encode('ascii')
                    if reconnects:
                        event('reconnect', error=last_error)
                    # A connection alone is not evidence of flowing frames.
                    first_frame = True
                    while True:
                        header, jpeg = receive(body, boundary)
                        receipt = stamp()
                        filename = f'frame-{frames:06d}-seq-{header["seq"]:010d}.jpg'
                        (output/filename).write_bytes(jpeg)
                        record = {'header': header, 'filename': filename, 'host_received_monotonic_ns': receipt,
                                  'sha256': hashlib.sha256(jpeg).hexdigest(), 'jpeg_validated': False,
                                  'connection_attempt': reconnects}
                        try:
                            from PIL import Image
                            with Image.open(io.BytesIO(jpeg)) as image:
                                header['width'], header['height'] = image.size
                            validate_jpeg(jpeg, header)
                            record['jpeg_validated'] = True
                        except Exception as exc:
                            raise ProtocolError(f'JPEG validation failed: {exc}') from exc
                        finally:
                            records.write(json.dumps(record)+'\n')
                            records.flush()
                        frames += 1
                        if first_frame:
                            status('streaming')
                            first_frame = False
                except TransportError as exc:
                    failures += 1
                    last_error = f'{type(exc).__name__}: {exc}'
                    event('disconnect', error=last_error)
                    if response is not None:
                        response.close()
                        response = None
                    check()
                    delay = min(5, 2 ** min(reconnects, 3))
                    retry_at = min(deadline, clock()+delay)
                    status('reconnecting', retry_at)
                    while clock() < retry_at:
                        check()
                        sleep(min(.2, retry_at-clock()))
                    check()
                    reconnects += 1
                    event('reconnect_attempt', error=last_error)
                    status('reconnecting')
                finally:
                    if response is not None:
                        response.close()
                        response = None
        except StopCapture:
            stopped = bool(stop_file and stop_file.exists())
        except KeyboardInterrupt:
            stopped = True
        except Exception as exc:
            failures += 1
            error = last_error = f'{type(exc).__name__}: {exc}'
            event('acquisition_error', error=error)
        finally:
            if response is not None:
                response.close()
            result = {'result': 'NO_VALID_FRAMES' if not frames else 'FRAMES_WITH_FAILURES' if failures else 'FRAMES_RECORDED', 'frames': frames, 'failure_count': failures, 'stopped_by_request': stopped,
                      'acquisition_error': error, 'last_error': last_error, 'reconnects': reconnects,
                      'elapsed_seconds': clock()-started, 'raw_bytes': raw.count,
                      'raw_sha256': raw.digest.hexdigest()}
            manifest['result'] = result
            (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
            (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
            status('stopped')
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
    return int(bool(result['failure_count']) or not result['frames'])


if __name__ == '__main__':
    raise SystemExit(main())
