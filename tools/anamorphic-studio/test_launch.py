"""Launcher process/network contracts. These tests never start a server."""
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import urllib.error

import launch


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='studio launcher test ')
        self.root = Path(self.temporary.name)
        (self.root / 'app' / 'node_modules').mkdir(parents=True)
        self.patches = [mock.patch.object(launch, 'ROOT', self.root),
                        mock.patch.object(launch.subprocess, 'Popen'),
                        mock.patch.object(launch.subprocess, 'run'),
                        mock.patch.object(launch.subprocess, 'call'),
                        mock.patch.object(launch.urllib.request, 'urlopen'),
                        mock.patch.object(launch.webbrowser, 'open'),
                        mock.patch.object(launch.time, 'sleep')]
        values = [patch.start() for patch in self.patches]
        self.process, self.run, self.call, self.urlopen, self.browser = values[1:6]
        self.process.return_value.pid = 12345
        self.process.return_value.poll.return_value = None

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.temporary.cleanup()

    def response(self, value):
        return io.BytesIO(json.dumps(value).encode())

    def test_posix_launch_detaches_and_disconnects_stdin(self):
        with mock.patch.object(launch, '_windows', return_value=False), \
                mock.patch.object(launch, 'running', side_effect=[False, False, True]):
            self.assertEqual(launch.main(['--no-browser']), 0)
        args, options = self.process.call_args
        self.assertEqual(args[0], [launch.sys.executable, 'tools/anamorphic-studio/server.py'])
        self.assertEqual(options['cwd'], self.root)
        self.assertTrue(options['start_new_session'])
        self.assertEqual(options['stdin'], subprocess.DEVNULL)
        self.assertEqual(options['stderr'], subprocess.STDOUT)
        self.assertNotIn('creationflags', options)
        self.run.assert_called_once_with(['npm', 'run', 'build'], cwd=self.root / 'app', check=True)
        self.browser.assert_not_called()
        self.assertEqual((self.root / 'out/anamorphic-studio/server.pid').read_text(), '12345')

    def test_windows_keeps_hidden_flags_and_command_arguments(self):
        (self.root / 'app/node_modules').rmdir()
        executable = str(self.root / 'Python with spaces' / 'python.exe')
        with mock.patch.object(launch, '_windows', return_value=True), \
                mock.patch.object(launch.sys, 'executable', executable), \
                mock.patch.object(launch, 'running', side_effect=[False, True]):
            self.assertEqual(launch.main(['--no-browser']), 0)
        self.assertEqual(self.process.call_args.args[0][0], executable)
        options = self.process.call_args.kwargs
        self.assertEqual(options['creationflags'], getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000))
        self.assertEqual(options['stdin'], subprocess.DEVNULL)
        self.assertNotIn('start_new_session', options)
        self.assertEqual(self.run.call_args_list, [mock.call(['npm.cmd', 'ci'], cwd=self.root / 'app', check=True),
                                                  mock.call(['npm.cmd', 'run', 'build'], cwd=self.root / 'app', check=True)])

    def test_posix_venv_reexec_uses_separate_arguments_for_spaces(self):
        executable = self.root / '.venv-studio/bin/python'
        executable.parent.mkdir(parents=True)
        executable.touch()
        self.call.return_value = 7
        with mock.patch.object(launch, '_windows', return_value=False), \
                mock.patch.object(launch.sys, 'executable', str(self.root / 'global python')):
            self.assertEqual(launch.main(['--no-browser', '--literal-argument']), 7)
        self.call.assert_called_once_with([str(executable), str(Path(launch.__file__).resolve()),
                                           '--no-browser', '--literal-argument'], cwd=self.root)
        self.process.assert_not_called()
        self.run.assert_not_called()
        self.urlopen.assert_not_called()

    def test_windows_venv_path_and_no_reexec_loop(self):
        executable = self.root / '.venv-studio/Scripts/python.exe'
        executable.parent.mkdir(parents=True)
        executable.touch()
        with mock.patch.object(launch, '_windows', return_value=True), \
                mock.patch.object(launch.sys, 'executable', str(executable)), \
                mock.patch.object(launch, 'running', return_value=True):
            self.assertEqual(launch.main(['--no-browser']), 0)
        self.call.assert_not_called()
        self.process.assert_not_called()

    def test_stop_only_skips_venv_build_browser_and_processes(self):
        executable = self.root / '.venv-studio/bin/python'
        executable.parent.mkdir(parents=True)
        executable.touch()
        self.urlopen.side_effect = [self.response({'app': 'Anamorphic Studio'}), self.response({'stopping': True})]
        self.assertEqual(launch.main(['--stop']), 0)
        request = self.urlopen.call_args.args[0]
        self.assertEqual(request.full_url, launch.URL + '/api/shutdown')
        self.assertEqual(request.get_method(), 'POST')
        self.assertEqual(request.get_header('X-studio-client'), 'anamorphic-studio')
        self.assertEqual(request.data, b'')
        self.assertEqual(self.urlopen.call_args.kwargs['timeout'], 15)
        self.call.assert_not_called()
        self.process.assert_not_called()
        self.run.assert_not_called()
        self.browser.assert_not_called()

    def test_stop_does_not_send_shutdown_to_other_app(self):
        self.urlopen.return_value = self.response({'app': 'Some other service'})
        self.assertEqual(launch.main(['--stop']), 0)
        self.urlopen.assert_called_once_with(launch.URL + '/api/status', timeout=2)
        self.process.assert_not_called()

    def test_stop_when_server_absent_has_no_side_effect(self):
        self.urlopen.side_effect = urllib.error.URLError('connection refused')
        self.assertEqual(launch.main(['--stop']), 0)
        self.urlopen.assert_called_once()
        self.call.assert_not_called()
        self.process.assert_not_called()
        self.run.assert_not_called()

    def test_failed_shutdown_is_not_reported_as_success(self):
        self.urlopen.side_effect = [self.response({'app': 'Anamorphic Studio'}), urllib.error.URLError('request failed')]
        with self.assertRaisesRegex(RuntimeError, 'Could not stop'):
            launch.main(['--stop'])

    def test_existing_server_only_opens_browser(self):
        with mock.patch.object(launch, 'running', return_value=True):
            self.assertEqual(launch.main([]), 0)
        self.browser.assert_called_once_with(launch.URL)
        self.run.assert_not_called()
        self.process.assert_not_called()

    def test_failed_child_reports_log_without_touching_other_processes(self):
        self.process.return_value.poll.return_value = 1
        with mock.patch.object(launch, '_windows', return_value=False), \
                mock.patch.object(launch, 'running', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'could not start'):
                launch.main(['--no-browser'])
        self.browser.assert_not_called()
        self.process.return_value.kill.assert_not_called()
        self.process.return_value.terminate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
