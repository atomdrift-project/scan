"""Exercise the benchmark runner without scanning real samples."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).resolve().parent / 'bench-balanced.py'


@unittest.skipUnless(sys.platform == 'linux', 'Linux helper affinity')
class BalancedBenchmarkTest(unittest.TestCase):
    def test_training_interrupt_flushes_but_never_passes(self):
        with tempfile.TemporaryDirectory(prefix='balanced-interrupt-') as tmp:
            root = Path(tmp)
            sample = root / 'sample.bin'
            sample.write_bytes(b'test')
            marker = root / 'flushed'
            scanner = root / 'scanner'
            scanner.write_text(
                f'#!{sys.executable}\nimport atexit,signal,sys,time\n'
                'from pathlib import Path\n'
                f'atexit.register(lambda: Path({str(marker)!r}).write_text("flushed"))\n'
                'signal.signal(signal.SIGINT, lambda *args: sys.exit(130))\n'
                'print("{}", flush=True)\ntime.sleep(60)\n'
            )
            scanner.chmod(0o755)
            prefix = root / 'result'
            result = subprocess.run(
                [sys.executable, str(RUNNER), str(sample), '--binary', str(scanner),
                 '--interrupt-after', '0.5', '--output', str(prefix)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            measurement = json.loads(Path(f'{prefix}.time.json').read_text())
            self.assertTrue(measurement['interrupted'])
            self.assertEqual(measurement['output_records'], 1)
            self.assertEqual(measurement['exit_code'], 130)
            self.assertEqual(marker.read_text(), 'flushed')

    def test_hostile_exit_is_success_only_with_complete_output(self):
        with tempfile.TemporaryDirectory(prefix='balanced-status-') as tmp:
            root = Path(tmp)
            sample = root / 'sample.bin'
            sample.write_bytes(b'test')
            scanner = root / 'scanner'
            cases = [
                ('hostile', '{}', 1, 0),
                ('error', '{"err":"analysis failed"}', 0, 1),
                ('skipped', '{"scanned":false}', 0, 1),
                ('missing', '', 0, 1),
            ]
            for name, output, scanner_exit, expected in cases:
                with self.subTest(name=name):
                    scanner.write_text(
                        f'#!{sys.executable}\nimport sys\n'
                        f'print({output!r})\nsys.exit({scanner_exit})\n'
                    )
                    scanner.chmod(0o755)
                    result = subprocess.run(
                        [sys.executable, str(RUNNER), str(sample), '--binary', str(scanner),
                         '--output', str(root / name)], capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, expected, result.stderr)

    def test_helper_environment_and_failed_series(self):
        with tempfile.TemporaryDirectory(prefix='balanced-harness-') as tmp:
            root = Path(tmp)
            corpus = root / 'corpus'
            corpus.mkdir()
            with (corpus / 'large.bin').open('wb') as sample:
                sample.truncate(17 * 1024 * 1024)
            helper = root / 'fake rizin'
            helper.write_text(
                f'#!{sys.executable}\nimport os,json\n'
                'print(json.dumps(sorted(os.sched_getaffinity(0))))\n'
            )
            helper.chmod(0o755)
            scanner = root / 'scanner'
            scanner.write_text(
                f'#!{sys.executable}\nimport os,json,subprocess,sys\n'
                'from pathlib import Path\n'
                'p=next(Path(sys.argv[-1]).iterdir())\n'
                'r=subprocess.check_output(["rizin",str(p)],'
                'env={"PATH":os.environ["PATH"]},text=True)\n'
                'print(json.dumps(dict(scanner_affinity=sorted(os.sched_getaffinity(0)),'
                'helper_affinity=json.loads(r))))\n'
            )
            scanner.chmod(0o755)
            core = min(os.sched_getaffinity(0))
            topology = Path(f'/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list')
            siblings = set()
            for part in topology.read_text().strip().split(','):
                bounds = part.split('-')
                siblings.update(range(int(bounds[0]), int(bounds[-1]) + 1))
            if not os.sched_getaffinity(0) - siblings:
                self.skipTest('requires a second physical core')
            prefix = root / 'result'
            result = subprocess.run(
                [sys.executable, str(RUNNER), str(corpus), '--binary', str(scanner),
                 '--rizin', str(helper), '--native-cpus', str(core), '--runs', '2',
                 '--limit', '0.000001', '--output', str(prefix)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1, (result.stdout, result.stderr))
            summary = json.loads(Path(f'{prefix}.series.json').read_text())
            self.assertEqual(len(summary['measurements']), 2)
            self.assertFalse(summary['all_passed'])
            for run in (1, 2):
                record = json.loads(Path(f'{prefix}.run{run}.jsonl').read_text())
                self.assertEqual(record['helper_affinity'], [core])
                self.assertTrue(siblings.isdisjoint(record['scanner_affinity']))
                self.assertTrue(Path(f'{prefix}.run{run}.rizin/starts.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
