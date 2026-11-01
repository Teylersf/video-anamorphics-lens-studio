"""Build and launch the local Studio; never deploys. GPL-2.0."""
from pathlib import Path
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import json
import webbrowser

ROOT = Path(__file__).resolve().parents[2]
URL = 'http://127.0.0.1:8766'


def running():
    try:
        with urllib.request.urlopen(URL + '/api/status', timeout=2) as response:
            return json.load(response).get('app') == 'Anamorphic Studio'
    except (OSError, ValueError):
        return False


def _windows():
    return os.name == 'nt'


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Shutdown only needs the standard library. It must not install/build the
    # app, launch an interpreter/server, or signal a terminal's process group.
    if '--stop' in arguments:
        if not running():
            print('Anamorphic Studio is not running.')
            return 0
        request = urllib.request.Request(URL + '/api/shutdown', data=b'', method='POST',
                                         headers={'X-Studio-Client': 'anamorphic-studio'})
        try:
            with urllib.request.urlopen(request, timeout=15):
                print('Anamorphic Studio is stopping.')
        except OSError as error:
            raise RuntimeError('Could not stop Anamorphic Studio through its local shutdown endpoint.') from error
        return 0
    local_python = ROOT / '.venv-studio' / ('Scripts/python.exe' if _windows() else 'bin/python')
    # Compare interpreter names without resolving symlinks: a POSIX venv's
    # python may point to the same binary while selecting a different sys.prefix.
    if local_python.exists() and Path(sys.executable).absolute() != local_python.absolute():
        return subprocess.call([str(local_python), str(Path(__file__).resolve()), *arguments], cwd=ROOT)
    if not running():
        npm = 'npm.cmd' if _windows() else 'npm'
        if not (ROOT / 'app/node_modules').is_dir():
            subprocess.run([npm, 'ci'], cwd=ROOT / 'app', check=True)
        subprocess.run([npm, 'run', 'build'], cwd=ROOT / 'app', check=True)
        logs = ROOT / 'out/anamorphic-studio'
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / 'server.log').open('ab') as log:
            process_options = {'stdin': subprocess.DEVNULL}
            if _windows():
                process_options['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
            else:
                process_options['start_new_session'] = True
            process = subprocess.Popen([sys.executable, 'tools/anamorphic-studio/server.py'], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       **process_options)
        (logs / 'server.pid').write_text(str(process.pid))
        for _ in range(120):
            if running():
                break
            if process.poll() is not None:
                raise RuntimeError('Studio could not start. See out/anamorphic-studio/server.log.')
            time.sleep(.5)
        else:
            raise RuntimeError('Studio is still starting. See out/anamorphic-studio/server.log.')
    print(URL)
    if '--no-browser' not in arguments:
        webbrowser.open(URL)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
