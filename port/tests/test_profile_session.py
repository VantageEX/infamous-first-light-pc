import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import paths  # noqa: F401  (puts scripts/ on sys.path)
import profile_session

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'launcher'))
from fl_launcher import profiling  # noqa: E402


class ProfileSessionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_no_session_without_the_switch(self):
        probe = self.dir / 'tlg-probe-v2.exe'
        probe.write_bytes(b'')
        self.assertEqual(profile_session.start({}, self.dir, probe), (probe, None, None))
        self.assertIsNone(profile_session.time_limit({'TLG_SESSION_SECONDS': '60'}, None))

    def test_session_without_the_profiling_build(self):
        probe = self.dir / 'tlg-probe-v2.exe'
        probe.write_bytes(b'')
        env = {'TLG_SESSION_DIR': str(self.dir / 's'), 'ISS_TIMELINE': '0', 'TLG_SESSION_SECONDS': '90'}
        used, capture, folder = profile_session.start(env, self.dir, probe)
        self.assertEqual((used, capture), (probe, None))  # sensors and timeline only
        self.assertEqual(env['ISS_TIMELINE'], '1')        # gpu_v2/defaults.env has it off
        self.assertEqual(env['ISS_TIMELINE_LOG'], str(folder / 'timeline.csv'))
        self.assertNotIn('TRACY_NO_EXIT', env)
        self.assertEqual(json.loads((folder / 'session.json').read_text())['probe'], 'tlg-probe-v2.exe')
        self.assertEqual(profile_session.time_limit(env, folder), 90.0)

    def test_profiling_build_without_capture(self):
        probe = self.dir / 'tlg-probe-v2.exe'
        probe.write_bytes(b'')
        (self.dir / 'tlg-probe-v2-profile.exe').write_bytes(b'')
        env = {'TLG_SESSION_DIR': str(self.dir / 's'), 'TLG_SESSION_CAPTURE': '0'}
        used, capture, _ = profile_session.start(env, self.dir, probe)
        self.assertEqual((used.name, capture), ('tlg-probe-v2-profile.exe', None))

    def test_launcher_benchmark_environment(self):
        routes = profiling.routes_dir(self.dir)
        routes.mkdir()
        (routes / 'route-a.txt').write_text('0 0 128 128 128 128 0 0\n')
        env, session = profiling.start_env(self.dir, True, 'route-a.txt', '120', False, stamp='20261026-120000')
        self.assertEqual(session.name, '20261026-120000-benchmark')
        self.assertEqual(env['TLG_SESSION_DIR'], str(session))
        self.assertEqual((env['TLG_SESSION_SECONDS'], env['TLG_PAD_REPLAY_AUTO']), ('120', '1'))
        self.assertEqual((session / 'route.txt').read_text(), (routes / 'route-a.txt').read_text())
        self.assertEqual(profiling.list_routes(self.dir), ['route-a.txt'])

    def test_launcher_route_recording_and_plain_start(self):
        env, session = profiling.start_env(self.dir, False, 'route-a.txt', '60', True, stamp='s')
        self.assertIsNone(session)
        self.assertEqual(env['TLG_PAD_RECORD_SECONDS'], '60')
        self.assertTrue(env['TLG_PAD_RECORD'].endswith('route-s.txt'))
        self.assertEqual(profiling.start_env(self.dir, False, '', '0', False), ({}, None))

    def test_pack_adds_the_log(self):
        session = profiling.sessions_dir(self.dir) / 'one'
        session.mkdir(parents=True)
        (session / 'sensors.csv').write_text('time_ns,name,value\n')
        (session / 'session.tracy').write_bytes(b'x' * 10)
        log = self.dir / 'run.log'
        log.write_text('log')
        archive = profiling.pack(session, log)
        with zipfile.ZipFile(archive) as z:
            self.assertEqual(sorted(z.namelist()), ['one/run.log', 'one/sensors.csv', 'one/session.tracy'])
            self.assertEqual(z.getinfo('one/session.tracy').compress_type, zipfile.ZIP_STORED)


if __name__ == '__main__':
    unittest.main()
