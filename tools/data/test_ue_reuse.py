"""Admission contracts for consumed UE inputs, without opening payloads."""
import contextlib
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asset_catalog as catalog
import asset_runtime as runtime
import ue_reuse


class UEReuseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ue-reuse-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts = self.root / 'artifacts.local'
        self.database = self.artifacts / catalog.DEFAULT_DATABASE_RELATIVE
        self.policy = self.root / 'ue-policy.json'
        self.contracts = []
        for locator in ('work/core', 'work/alias', 'work/reserved'):
            folder = self.artifacts / locator
            (folder / 'observations').mkdir(parents=True)
            (folder / 'observations/frame.npz').write_bytes(b'not decoded by preflight')
            (folder / 'truth.json').write_text('evaluator payload', encoding='utf-8')
            self.contracts.append(dict(
                locator=locator, source_family='core-source',
                evidence_status='development_consumed', evidence='evidence.md',
                allowed_inputs=[dict(relative_path='observations', role='observation'),
                                dict(relative_path='truth.json', role='evaluator')]))
        self.contracts[-1].update(evidence_status='reserved', blocked_reason='Test unactivated')
        (self.root / 'evidence.md').write_text('Consumed Development; test reserved.', encoding='utf-8')
        rule = dict(asset_kind='dataset', asset_class='data', evidence_status='development_consumed',
                    storage_status='shared', owner='fixture', retention_reason='contract test')
        discovery = self.root / 'catalog-policy.json'
        discovery.write_text(json.dumps(dict(schema='blindassist-asset-management-policy-v1',
            catalog_unit='direct_child', roots={'work': rule}, managed_assets={},
            excluded_roots={'evidence': 'catalog storage'}, top_level_file_exclude_globs=[],
            fallback=rule)), encoding='utf-8')
        self.discover()

    def discover(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = catalog.main(['discover', '--artifact-root', str(self.artifacts),
                '--database', str(self.database), '--policy', str(self.root / 'catalog-policy.json'), '--repo-root', str(self.root)])
        self.assertEqual(code, 0)

    def check_spec(self, inputs=None, **changes):
        self.policy.write_text(json.dumps(dict(schema='blindassist-ue-reuse-policy-v1',
            contracts=self.contracts)), encoding='utf-8')
        spec = dict(id='fixture', reuse=dict(mode='regression', query='core observations'),
            inputs=inputs or [dict(alias='obs', asset='work/core',
                                  relative_path='observations', role='observation')])
        spec.update(changes)
        return ue_reuse.preflight(spec, artifact_root=self.artifacts, repo_root=self.root,
            database=self.database, policy_path=self.policy, required=True)

    def blocked(self, result):
        self.assertEqual(result['status'], 'BLOCKED', result)
        self.assertTrue(result['errors'])

    def test_consumed_observation_admitted(self):
        result = self.check_spec()
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(len(result['inputs']), 1)

    def test_missing_reuse_and_fresh_mode_denied(self):
        self.blocked(self.check_spec(reuse=None))
        self.blocked(self.check_spec(reuse=dict(mode='fresh', query='core')))

    def test_reserved_and_mixed_root_not_unlocked_by_split(self):
        for locator in ('work/reserved', 'work/core'):
            with self.subTest(locator=locator):
                self.blocked(self.check_spec([dict(alias='obs', asset=locator,
                    role='observation', split='train')]))
        self.blocked(self.check_spec([dict(alias='obs', asset='work/reserved',
            relative_path='observations', role='observation', split='train')]))

    def test_role_is_enforced_for_asset_and_raw_path(self):
        for item in (dict(asset='work/core', relative_path='truth.json'),
                     dict(path=str(self.artifacts / 'work/core/truth.json'))):
            with self.subTest(item=item):
                self.blocked(self.check_spec([dict(alias='obs', role='observation', **item)]))
        result = self.check_spec([dict(alias='truth', asset='work/core',
                                      relative_path='truth.json', role='evaluator')])
        self.assertEqual(result['status'], 'PASS', result)

    def test_raw_observation_path_admitted_but_escape_denied(self):
        result = self.check_spec([dict(alias='obs', role='observation',
            path=str(self.artifacts / 'work/core/observations/frame.npz'))])
        self.assertEqual(result['status'], 'PASS', result)
        outside = self.root / 'outside.npz'
        outside.write_bytes(b'outside')
        self.blocked(self.check_spec([dict(alias='obs', role='observation', path=str(outside))]))
        self.blocked(self.check_spec([dict(alias='obs', role='observation', asset='work/core',
                                         relative_path='../reserved/observations')]))

    def test_source_aliases_share_one_family(self):
        result = self.check_spec([dict(alias=alias, role='observation', asset='work/' + alias,
            relative_path='observations') for alias in ('core', 'alias')])
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(len(result['source_families']), 1)

    def test_unknown_contract_and_missing_registry_denied(self):
        self.contracts = []
        self.blocked(self.check_spec())
        self.database = self.root / 'missing.sqlite3'
        self.blocked(self.check_spec())
        self.assertFalse(self.database.exists())

    def test_blocked_preflight_does_not_record_usage(self):
        with contextlib.closing(sqlite3.connect(self.database)) as connection:
            before = connection.execute('SELECT COUNT(*) FROM usage_events').fetchone()[0]
        self.blocked(self.check_spec(reuse=None))
        with contextlib.closing(sqlite3.connect(self.database)) as connection:
            after = connection.execute('SELECT COUNT(*) FROM usage_events').fetchone()[0]
        self.assertEqual(before, after)

    def test_runtime_blocks_before_launch_or_consumption(self):
        self.check_spec()  # Materialize the reviewed fixture policy.
        marker = self.root / 'launched.txt'
        spec = dict(schema=runtime.RUN_SCHEMA, id='blocked-runtime', route='ue-reuse',
            question='Check reserved input admission', evaluator='fixture',
            evidence_boundary='Fixture only', reuse=dict(mode='regression', query='core'),
            command=[sys.executable, '-c', 'from pathlib import Path; Path(__import__("sys").argv[1]).touch()', str(marker)],
            inputs=[dict(alias='obs', asset='work/reserved', relative_path='observations', role='observation')],
            outputs=[])
        path = self.root / 'runtime.json'
        path.write_text(json.dumps(spec), encoding='utf-8')
        with contextlib.closing(sqlite3.connect(self.database)) as connection:
            before = connection.execute('SELECT COUNT(*) FROM usage_events').fetchone()[0]
        with patch.object(ue_reuse, 'DEFAULT_POLICY', self.policy):
            with self.assertRaisesRegex(runtime.RuntimeError_, 'UE reuse blocked'):
                runtime.run_spec(path, repo_root=self.root, artifact_root=self.artifacts,
                                 policy_path=self.root / 'catalog-policy.json')
        self.assertFalse(marker.exists())
        journal = json.loads(runtime.journal_path(self.artifacts, 'ue-reuse', 'blocked-runtime').read_text())
        self.assertEqual(journal['reuse_preflight']['status'], 'BLOCKED')
        with contextlib.closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(before, connection.execute('SELECT COUNT(*) FROM usage_events').fetchone()[0])

    def test_derived_lineage_survives_discovery_and_preserves_role(self):
        report = self.check_spec()
        output = self.artifacts / 'work/derived/predictions.json'
        output.parent.mkdir()
        output.write_text('{}', encoding='utf-8')
        self.discover()
        edges = ue_reuse.record_output_lineage(report, database=self.database,
            artifact_root=self.artifacts, outputs=[dict(path='work/derived/predictions.json')],
            run_id='fixture-derived', evaluator='fixture-replay')
        self.assertEqual(len(edges), 1)
        with contextlib.closing(sqlite3.connect(self.database)) as connection:
            sources = connection.execute('SELECT a.locator FROM derivation_inputs d '
                'JOIN assets a ON a.asset_id=d.input_asset_id WHERE d.derivation_id=?', (edges[0],)).fetchall()
            metadata = connection.execute("SELECT metadata_json FROM assets WHERE locator='work/derived'").fetchone()[0]
        self.assertEqual(sources, [('work/core',)])
        self.discover()
        with contextlib.closing(sqlite3.connect(self.database)) as connection:
            refreshed = connection.execute("SELECT metadata_json FROM assets WHERE locator='work/derived'").fetchone()[0]
        self.assertEqual(json.loads(metadata)['ue_reuse'], json.loads(refreshed)['ue_reuse'])
        item = dict(alias='prediction', asset='work/derived', relative_path='predictions.json', role='evaluator')
        admitted = self.check_spec([item])
        self.assertEqual(admitted['status'], 'PASS', admitted)
        self.assertEqual(admitted['source_families'], ['core-source'])
        self.assertIn('work/derived', [candidate['locator'] for candidate in admitted['candidates']])
        self.blocked(self.check_spec([dict(item, role='observation')]))
        self.blocked(self.check_spec([item], reuse=dict(mode='training', query='derived prediction')))

    def test_multiple_outputs_in_one_unit_keep_all_source_families(self):
        first = self.check_spec()
        self.contracts[1]['source_family'] = 'second-source'
        second = self.check_spec([dict(alias='obs', asset='work/alias',
            relative_path='observations', role='observation')])
        output = self.artifacts / 'work/derived'
        output.mkdir()
        for name in ('first.json', 'second.json'):
            (output / name).write_text('{}', encoding='utf-8')
        self.discover()
        for index, report in enumerate((first, second)):
            ue_reuse.record_output_lineage(report, database=self.database,
                artifact_root=self.artifacts, outputs=[dict(path=f'work/derived/{("first", "second")[index]}.json')],
                run_id=f'fixture-derived-{index}', evaluator='fixture-replay')
        admitted = self.check_spec([dict(alias='prediction', asset='work/derived',
            relative_path='first.json', role='evaluator')])
        self.assertEqual(admitted['source_families'], ['core-source', 'second-source'])


if __name__ == '__main__':
    unittest.main()
