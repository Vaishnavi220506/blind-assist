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
class Tests(unittest.TestCase):
 def test_chunk_decode_and_eof_preservation(self):
  data=fixture()
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'run';r=response(data);result=c.acquire('http://192.168.31.43:81/stream',out,2,opener=lambda *a,**k:r)
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
if __name__=='__main__': unittest.main()
