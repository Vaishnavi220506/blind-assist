import unittest
from pathlib import Path
import tempfile
import json
from ba_nfo_preflight import main

class NfoPreflightTests(unittest.TestCase):
    def test_writes_explicit_block_without_training(self):
        with tempfile.TemporaryDirectory() as d:
            main(Path(d)); report=json.loads((Path(d)/'preflight.json').read_text())
            self.assertEqual(report['status'],'BLOCKED_DATA_PREREQUISITES')
            self.assertIn('RGB+ToF+ordinal',report['frozen_protocol']['arms'])
            self.assertFalse((Path(d)/'checkpoint.pt').exists())

if __name__=='__main__':unittest.main()
