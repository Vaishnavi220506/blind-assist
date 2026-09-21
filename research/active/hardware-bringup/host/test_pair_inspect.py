"""Synthetic pairing fixtures; no device or motion evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from pair_inspect import RELATION, center_distances, inspect


def artifact_temp():
    root = Path(__file__).resolve().parents[4]/"artifacts.local"/"hardware-bringup"/"test-tmp"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="pair-inspect-", dir=root)


def tof(seq, time, valid=True):
    return {"host_received_monotonic_ns": time,
            "sensor": {"type": "cnh_frame", "seq": seq, "distance_mm": [600 if valid else 0]*16,
                       "target_status": [5]*16, "nb_target": [1]*16}}


def fixture(root, camera_times, tof_rows):
    (root/"camera").mkdir(parents=True)
    (root/"tof").mkdir()
    cameras = []
    for index, timestamp in enumerate(camera_times):
        filename = f"synthetic-{index}.jpg"
        (root/"camera"/filename).write_bytes(b"synthetic reference only; JPEG decoding owned by acquisition")
        cameras.append({"header": {"seq": index+1}, "host_received_monotonic_ns": timestamp,
                        "filename": filename, "jpeg_validated": True})
    for path, records in ((root/"camera"/"frames.jsonl", cameras), (root/"tof"/"frames.jsonl", tof_rows)):
        path.write_text("".join(json.dumps(record)+"\n" for record in records), encoding="utf-8")


class InspectTests(unittest.TestCase):
    def test_nearest_tie_earlier_and_overlap(self):
        with artifact_temp() as temp:
            source, out = Path(temp)/"paired", Path(temp)/"inspection"
            fixture(source, [100, 300, 500], [tof(1, 100), tof(2, 200), tof(3, 450), tof(4, 500)])
            summary = inspect(source, out)
            pairs = [json.loads(line) for line in (out/"pairs.jsonl").read_text().splitlines()]
            self.assertEqual([1, 1, 3, 3], [row["camera_seq"] for row in pairs])
            self.assertEqual(RELATION, summary["relation"])
            self.assertEqual(400/1e9, summary["overlap_seconds"])
            self.assertEqual(100/1e6, summary["absolute_receipt_delta_ms_max"])

    def test_nonoverlap_no_pair_still_retains_all_tof(self):
        with artifact_temp() as temp:
            source, out = Path(temp)/"paired", Path(temp)/"inspection"
            fixture(source, [100, 200], [tof(1, 300), tof(2, 400)])
            summary = inspect(source, out)
            self.assertEqual(0, summary["pairs"])
            self.assertEqual(0, summary["overlap_seconds"])
            self.assertIsNone(summary["absolute_receipt_delta_ms_median"])
            self.assertEqual(2, len((out/"tof_center_curve.jsonl").read_text().splitlines()))

    def test_unknown_distance_preserved_and_not_zero(self):
        item = tof(2, 200, valid=False)
        item["sensor"]["distance_mm"][5] = 700
        item["sensor"]["target_status"][5] = 9
        zones, values, median, error = center_distances(item)
        self.assertEqual([5, 6, 9, 10], zones)
        self.assertEqual([None]*4, values)
        self.assertIsNone(median)
        self.assertIsNone(error)
        with artifact_temp() as temp:
            source, out = Path(temp)/"paired", Path(temp)/"inspection"
            fixture(source, [100, 200, 300], [tof(1, 100), item, tof(3, 300)])
            summary = inspect(source, out)
            self.assertEqual(3, summary["pairs"])
            self.assertEqual(1, summary["tof_missing_center_distance_records"])
            records = [json.loads(line) for line in (out/"tof_records.jsonl").read_text().splitlines()]
            self.assertEqual(item, records[1]["record"])

    def test_invalid_record_retained(self):
        with artifact_temp() as temp:
            source, out = Path(temp)/"paired", Path(temp)/"inspection"
            fixture(source, [100], [tof(1, 100)])
            with (source/"tof"/"frames.jsonl").open("a") as stream:
                stream.write("BROKEN JSON\n")
            summary = inspect(source, out)
            self.assertEqual(2, summary["tof_records"])
            self.assertEqual(1, summary["tof_parse_errors"])
            records = [json.loads(line) for line in (out/"tof_records.jsonl").read_text().splitlines()]
            self.assertEqual("BROKEN JSON", records[1]["raw_line_if_invalid"])

    def test_overwrite_or_ancestor_refused(self):
        with artifact_temp() as temp:
            source = Path(temp)/"paired"
            fixture(source, [100], [tof(1, 100)])
            for out in (source, Path(temp), source/"camera"):
                with self.subTest(out=out), self.assertRaises((ValueError, FileExistsError)):
                    inspect(source, out)


if __name__ == "__main__":
    unittest.main()
