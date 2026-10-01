import hashlib
from http.client import HTTPResponse
import io,json
from pathlib import Path
import tempfile,unittest
from PIL import Image
import wifi_camera_capture as c
class Socket:
 def __init__(self,data): self.data=io.BytesIO(data)
 def makefile(self,*args): return self.data
def fixture():
 b=io.BytesIO();Image.new('RGB',(16,12)).save(b,'JPEG');j=b.getvalue()
 return b'\r\n--BAFRAME\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(j)).encode()+b'\r\nX-Sequence-Id: fixture\r\nX-Frame-Sequence: 4\r\nX-Capture-Timestamp-Us: 100\r\nX-Jpeg-Ready-Timestamp-Us: 120\r\nX-Device-Send-Start-Timestamp-Us: 130\r\n\r\n'+j
def response(data):
 chunks=[data[i:i+17] for i in range(0,len(data),17)]
 wire=b'HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace;boundary=BAFRAME\r\nTransfer-Encoding: chunked\r\n\r\n'+b''.join(f'{len(x):x}\r\n'.encode()+x+b'\r\n' for x in chunks)+b'0\r\n\r\n'
 r=HTTPResponse(Socket(wire));r.begin();return r
class Clock:
 def __init__(self): self.now=0.
 def __call__(self): return self.now
 def sleep(self,n): self.now+=n
class Tests(unittest.TestCase):
 def test_chunk_decode_and_eof_preservation(self):
  data=fixture()
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'run';r=response(data);timer=Clock();result=c.acquire('http://192.168.31.43:81/stream',out,.5,opener=lambda *a,**k:r,clock=timer,sleep=timer.sleep)
   self.assertEqual(result['frames'],1);self.assertEqual(result['failure_count'],1)
   self.assertEqual((out/'raw.bin').read_bytes(),data);self.assertEqual(result['raw_sha256'],hashlib.sha256(data).hexdigest())
   record=json.loads((out/'frames.jsonl').read_text());self.assertTrue(record['jpeg_validated']);self.assertEqual(record['header']['capture_timestamp_us'],100);self.assertTrue(r.isclosed())
 def test_malformed_cap(self):
  data=b'\r\n--BAFRAME\r\n'+b'A'*10000
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'run';result=c.acquire('http://192.168.31.43:81/stream',out,2,opener=lambda *a,**k:response(data))
   raw=(out/'raw.bin').read_bytes();self.assertEqual(result['failure_count'],1);self.assertGreaterEqual(len(raw),c.MAX_HEADERS);self.assertTrue(data.startswith(raw));self.assertEqual(result['raw_sha256'],hashlib.sha256(raw).hexdigest())
 def test_stop_preserves_received_partial(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'run';stop=Path(tmp)/'stop';r=response(fixture());read=r.read1
   def read_stop(n):
    data=read(n);stop.touch();return data
   r.read1=read_stop
   result=c.acquire('http://192.168.31.43:81/stream',out,2,stop,opener=lambda *a,**k:r)
   self.assertTrue(result['stopped_by_request']);self.assertEqual(result['failure_count'],0);self.assertGreater((out/'raw.bin').stat().st_size,0);self.assertTrue(r.isclosed())
 def test_url(self):
  self.assertEqual(c.validate_url('http://192.168.31.43:81/stream'),'http://192.168.31.43:81/stream')
  for url in ['http://example.com:81/stream','http://8.8.8.8:81/stream','http://127.0.0.1:81/stream','http://192.168.1.2/stream','http://192.168.1.2:81/stream?x=1','http://user@192.168.1.2:81/stream']:
   with self.assertRaises(ValueError): c.validate_url(url)
 def test_reconnect_preserves_raw_and_frame_names(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'run';timer=Clock();data=fixture();responses=[]
   def opener(*a,**k):
    r=response(data);responses.append(r);return r
   result=c.acquire('http://192.168.31.43:81/stream',out,1.5,opener=opener,clock=timer,sleep=timer.sleep)
   self.assertEqual(result['frames'],2);self.assertEqual(result['reconnects'],1);self.assertEqual(result['failure_count'],2)
   self.assertIsNone(result['acquisition_error']);self.assertEqual(result['elapsed_seconds'],1.5)
   self.assertEqual((out/'raw.bin').read_bytes(),data+data);self.assertEqual(result['raw_sha256'],hashlib.sha256(data+data).hexdigest())
   rows=[json.loads(x) for x in (out/'frames.jsonl').read_text().splitlines()];self.assertNotEqual(rows[0]['filename'],rows[1]['filename'])
   self.assertTrue(all(r.isclosed() for r in responses));self.assertEqual(json.loads((out/'status.json').read_text())['state'],'stopped')
   kinds=[json.loads(x)['kind'] for x in (out/'events.jsonl').read_text().splitlines()];self.assertIn('disconnect',kinds);self.assertIn('reconnect',kinds)
 def test_stop_during_backoff(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'run';stop=Path(tmp)/'stop';timer=Clock();calls=[]
   def opener(*a,**k): calls.append(1);raise TimeoutError('fixture disconnected')
   def sleep(n):
    self.assertLessEqual(n,.2);timer.sleep(n);stop.touch()
   result=c.acquire('http://192.168.31.43:81/stream',out,30,stop,opener=opener,clock=timer,sleep=sleep)
   self.assertTrue(result['stopped_by_request']);self.assertEqual(len(calls),1);self.assertEqual(result['elapsed_seconds'],.2);self.assertEqual(result['failure_count'],1)
 def test_network_error_then_recovery(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'run';timer=Clock();calls=[]
   def opener(*a,**k):
    calls.append(1)
    if len(calls)==1: raise ConnectionResetError('fixture reset')
    return response(fixture())
   result=c.acquire('http://192.168.31.43:81/stream',out,1.5,opener=opener,clock=timer,sleep=timer.sleep)
   self.assertEqual(result['frames'],1);self.assertEqual(result['reconnects'],1);self.assertEqual(result['elapsed_seconds'],1.5)
 def test_corrupt_jpeg_not_retried(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'run';data=fixture().replace(b'\xff\xd8',b'xx',1);calls=[]
   def opener(*a,**k): calls.append(1);return response(data)
   result=c.acquire('http://192.168.31.43:81/stream',out,2,opener=opener)
   self.assertEqual(len(calls),1);self.assertEqual(result['reconnects'],0);self.assertIn('JPEG validation failed',result['acquisition_error']);self.assertEqual((out/'raw.bin').read_bytes(),data)
if __name__=='__main__': unittest.main()
