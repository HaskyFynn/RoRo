"""Build a clean upload ZIP and SHA256 manifest without trial data or Git internals."""
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
EXCLUDED = {'.git', '.venv', '__pycache__', 'runs', 'demo_runs', 'snapshots', 'targets', 'verification-artifacts'}


def main():
    files = []
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED for part in relative.parts):
            continue
        if path.is_file() and path.name not in {'local_setup.json', 'MANIFEST.sha256'} and path.suffix not in {'.pyc', '.tmp', '.zip'}:
            files.append(path)
    files.sort()
    manifest = ROOT / 'MANIFEST.sha256'
    manifest.write_text(''.join(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}\n' for path in files), encoding='utf-8')
    output = ROOT.parent / 'roemotion-experiments-upload.zip'
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files + [manifest]:
            archive.write(path, 'roemotion-experiments/' + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for path in files:
            name = 'roemotion-experiments/' + path.relative_to(ROOT).as_posix()
            assert hashlib.sha256(archive.read(name)).digest() == hashlib.sha256(path.read_bytes()).digest()
    print(f'{output}\n{len(files)+1} files; {output.stat().st_size/1_000_000:.2f} MB; archive checksums verified')


if __name__ == '__main__':
    main()
