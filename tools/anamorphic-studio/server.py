"""Loopback-only Anamorphic Studio service. Original media stays local. GPL-2.0."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid
from typing import Any

import av
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / 'out/anamorphic-studio'
STORE.mkdir(parents=True, exist_ok=True)
GPU_LOCK = threading.Lock()
STATE_LOCK = threading.RLock()
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='studio-export')
MEDIA: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
CANCEL: dict[str, threading.Event] = {}
SERVER = None
VERSION = '0.2.0-local'
MAX_UPLOAD = 64 * 1024**3
EXTENSIONS = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.mxf', '.m4v', '.mts', '.m2ts', '.mpg', '.mpeg', '.ts', '.m2v', '.wmv', '.flv', '.ogv'}


def pipeline():
    import media_pipeline
    return media_pipeline


def engine():
    import studio_engine
    return studio_engine


def validated_settings(value):
    try:
        return engine().normalize_settings(value)
    except (TypeError, ValueError) as error:
        raise HTTPException(422, str(error)) from error


def save_json(path, data):
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def public_job(job):
    return {key: value for key, value in job.items() if not key.startswith('_')}


def update_job(identifier, **values):
    with STATE_LOCK:
        job = JOBS[identifier]
        job.update(values)
        save_json(STORE / 'jobs' / identifier / 'job.json', public_job(job))
        return public_job(job)


def get_media(identifier):
    with STATE_LOCK:
        value = MEDIA.get(identifier)
    if not value:
        raise HTTPException(404, 'Video not found. Import it again.')
    return value


def source_path(media):
    return STORE / 'media' / media['id'] / ('source' + media['container'])


@asynccontextmanager
async def lifespan(app):
    for folder in (STORE / 'media', STORE / 'jobs'):
        folder.mkdir(exist_ok=True)
    for path in (STORE / 'media').glob('*/media.json'):
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            if uuid.UUID(value['id']).hex != value['id']:
                continue
            if source_path(value).is_file():
                MEDIA[value['id']] = value
        except (ValueError, KeyError, OSError):
            continue
    for path in (STORE / 'jobs').glob('*/job.json'):
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            if value['status'] in ('queued', 'running', 'cancelling'):
                value.update(status='interrupted', message='The service restarted. Start a new export.')
            JOBS[value['id']] = value
        except (ValueError, KeyError, OSError):
            continue
    yield
    for event in CANCEL.values():
        event.set()
    EXECUTOR.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title='Anamorphic Studio', version=VERSION, lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', '[::1]', 'testserver'])
app.add_middleware(CORSMiddleware, allow_origins=[
    'http://127.0.0.1:5173', 'http://localhost:5173', 'http://127.0.0.1:8766', 'http://localhost:8766'
], allow_methods=['GET', 'POST'], allow_headers=['Content-Type', 'X-Studio-Client'])


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError):
    # Do not echo non-finite input values into a JSON error response.
    return JSONResponse(status_code=422, content={'detail': [
        {key: item[key] for key in ('loc', 'msg', 'type')} for item in error.errors()
    ]})


@app.middleware('http')
async def local_mutations(request: Request, call_next):
    if request.method == 'POST' and request.headers.get('X-Studio-Client') != 'anamorphic-studio':
        return JSONResponse({'detail': 'Use the local Studio app to make this request.'}, status_code=403)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.get('/api/status')
def status():
    import torch
    ready = torch.cuda.is_available() and bool(shutil.which('ffmpeg')) and bool(shutil.which('ffprobe'))
    return {'app': 'Anamorphic Studio', 'version': VERSION, 'ready': ready,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable',
            'ffmpeg': bool(shutil.which('ffmpeg')), 'local_only': True, 'gpu_busy': GPU_LOCK.locked()}


@app.get('/api/config')
def config():
    core = engine()
    return {'defaults': core.DEFAULTS, 'profiles': core.PROFILES, 'max_dimension': 8192,
            'max_pixels': 35_651_584, 'default_export': 'match'}


@app.post('/api/media')
def import_media(file: UploadFile):
    name = Path((file.filename or 'video.mp4').replace('\\', '/')).name
    extension = Path(name).suffix.lower()
    if extension not in EXTENSIONS:
        raise HTTPException(415, 'Choose a video file such as MOV, MP4, MKV, MXF, AVI or WebM.')
    identifier = uuid.uuid4().hex
    folder = STORE / 'media' / identifier
    folder.mkdir(parents=True)
    original = folder / ('source' + extension)
    length = 0
    try:
        with original.open('xb') as target:
            while block := file.file.read(4 * 1024 * 1024):
                length += len(block)
                if length > MAX_UPLOAD:
                    raise ValueError('This local import exceeds the 64 GiB file limit.')
                if shutil.disk_usage(STORE).free < len(block) + 512 * 1024**2:
                    raise ValueError('Insufficient local storage for this import.')
                target.write(block)
        if not length:
            raise ValueError('The selected video is empty.')
        details = pipeline().inspect_media(original)
        width, height = int(details['width']), int(details['height'])
        if max(width, height) > 8192 or width * height > 35_651_584:
            raise ValueError('This source exceeds the 8K processing envelope (8192 per axis, 35.65 MP).')
        pipeline().make_proxy(original, folder / 'proxy.mp4', folder / 'poster.jpg')
        details.update(id=identifier, name=name, container=extension, bytes=length,
                       preview_url=f'/api/media/{identifier}/proxy.mp4', poster_url=f'/api/media/{identifier}/poster.jpg')
        details.setdefault('warnings', [])
        save_json(folder / 'media.json', details)
        with STATE_LOCK:
            MEDIA[identifier] = details
        return details
    except Exception as error:
        # Only the newly allocated upload directory belongs to this failed run.
        if folder.resolve().parent != (STORE / 'media').resolve():
            raise RuntimeError('Refusing to remove an unexpected upload directory') from error
        shutil.rmtree(folder)
        raise HTTPException(422, str(error)) from error
    finally:
        file.file.close()


@app.get('/api/media/{identifier}')
def media_details(identifier: str):
    return get_media(identifier)


@app.get('/api/media/{identifier}/{asset}')
def media_asset(identifier: str, asset: str):
    get_media(identifier)
    if asset not in ('proxy.mp4', 'poster.jpg'):
        raise HTTPException(404, 'Unknown asset')
    path = STORE / 'media' / identifier / asset
    if not path.is_file():
        raise HTTPException(404, 'Asset unavailable')
    return FileResponse(path, media_type='video/mp4' if asset.endswith('.mp4') else 'image/jpeg')


class PreviewRequest(BaseModel):
    media_id: str
    time: float = Field(default=0, ge=0)
    settings: dict[str, Any] = Field(default_factory=dict)
    width: int = Field(default=1280, ge=320, le=1920)


def frame_at(path, requested_time, metadata):
    import numpy as np
    with av.open(str(path), options={'protocol_whitelist': 'file,pipe'}) as container:
        selected = metadata.get('video_stream_index')
        video = container.streams[selected] if isinstance(selected, int) else next(
            stream for stream in container.streams.video if not (stream.disposition & 1024))
        start = float((video.start_time or 0) * video.time_base)
        container.seek(int((requested_time + start) / float(video.time_base)), stream=video, backward=True)
        chosen = None
        timestamp = 0.0
        for frame in container.decode(video):
            chosen = frame
            timestamp = float(frame.pts * frame.time_base) - start if frame.pts is not None else requested_time
            if timestamp >= requested_time - .001:
                break
        if chosen is None:
            raise ValueError('No video frame could be decoded at that position.')
        matrix = chosen.colorspace
        if matrix in (0, 2):
            matrix = 1 if chosen.height >= 720 else 5
        source_range = chosen.color_range or (2 if str(chosen.format.name).startswith(('rgb', 'gbr', 'yuvj')) else 1)
        rgb_frame = av.video.reformatter.VideoReformatter().reformat(
            chosen, format='rgb48le', src_colorspace=matrix,
            src_color_range=source_range, dst_color_range=2)
        rgb = rgb_frame.to_ndarray()
    rotation = int(metadata.get('rotation', 0))
    if rotation % 90 == 0 and rotation:
        rgb = np.rot90(rgb, k=round(rotation / 90)).copy()
    return rgb, max(0.0, timestamp)


@app.post('/api/preview')
def preview(request: PreviewRequest):
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    media = get_media(request.media_id)
    if not math.isfinite(request.time):
        raise HTTPException(422, 'Invalid preview position')
    settings = validated_settings(request.settings)
    position = min(request.time, max(0, float(media['duration']) - .001))
    cache_key = hashlib.sha256(json.dumps([position, settings, request.width], sort_keys=True).encode()).hexdigest()[:24]
    folder = STORE / 'media' / media['id'] / 'frames'
    folder.mkdir(exist_ok=True)
    before, after = folder / (cache_key + '-before.jpg'), folder / (cache_key + '-after.jpg')
    info_path = folder / (cache_key + '.json')
    if before.exists() and after.exists() and info_path.exists():
        return json.loads(info_path.read_text())
    if not GPU_LOCK.acquire(blocking=False):
        raise HTTPException(409, 'The GPU is rendering an export. Existing previews remain available.')
    started = time.perf_counter()
    try:
        array, actual_time = frame_at(source_path(media), position, media)
        height, width = array.shape[:2]
        tensor = torch.from_numpy(array.astype(np.float32)).to('cuda').permute(2, 0, 1) / 65535
        scale = min(1, request.width / width, 1080 / height)
        if scale < 1:
            tensor = F.interpolate(tensor[None], size=(max(1, round(height * scale)), max(1, round(width * scale))),
                                   mode='bilinear', align_corners=False, antialias=True)[0]
        transfer = str(media.get('transfer') or 'bt709')
        rendered = engine().StudioRenderer(settings, transfer=transfer).render(
            tensor, frame_index=round(actual_time * float(media['fps'])), time_s=actual_time)
        primaries = str(media.get('color_primaries') or 'bt709')
        for frame, target in ((tensor, before), (rendered, after)):
            display = engine().to_display_rgb(frame, transfer, primaries)
            pixels = (display.clamp(0, 1).permute(1, 2, 0).contiguous().cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(pixels).save(target, format='JPEG', quality=94, subsampling=0)
        info = {'original_url': f'/api/frames/{media["id"]}/{before.name}',
                'processed_url': f'/api/frames/{media["id"]}/{after.name}', 'time': actual_time,
                'width': int(rendered.shape[2]), 'height': int(rendered.shape[1]),
                'render_ms': round((time.perf_counter() - started) * 1000),
                'hdr_display_preview': transfer in ('pq', 'hlg', 'smpte2084', 'arib-std-b67')}
        save_json(info_path, info)
        del rendered, tensor
        return info
    except Exception as error:
        raise HTTPException(422, str(error)) from error
    finally:
        torch.cuda.empty_cache()
        GPU_LOCK.release()


@app.get('/api/frames/{identifier}/{name}')
def frame_asset(identifier: str, name: str):
    get_media(identifier)
    if not re.fullmatch(r'[0-9a-f]{24}-(before|after)\.jpg', name):
        raise HTTPException(404, 'Unknown preview')
    path = STORE / 'media' / identifier / 'frames' / name
    if not path.is_file():
        raise HTTPException(404, 'Preview unavailable')
    return FileResponse(path, media_type='image/jpeg')


class JobRequest(BaseModel):
    media_id: str
    settings: dict[str, Any] = Field(default_factory=dict)
    mode: str = 'match'
    preview: bool = False


def execute_job(identifier, media, settings, mode, is_preview):
    event = CANCEL[identifier]
    acquired = False
    try:
        while not event.is_set():
            if GPU_LOCK.acquire(timeout=.2):
                acquired = True
                break
        if event.is_set():
            update_job(identifier, status='cancelled', message='Cancelled before rendering')
            return
        update_job(identifier, status='running', message='Preparing source-matched render', progress=0)
        extension = '.mp4' if is_preview or mode == 'hevc' else '.mkv' if mode == 'lossless' else '.mov' if mode == 'prores' else media['container']
        name = Path(media['name']).stem + ('-preview' if is_preview else '-anamorphic') + extension
        output = STORE / 'jobs' / identifier / name
        def progress(value, message):
            update_job(identifier, progress=max(0, min(.99, float(value))), message=str(message))
        result = pipeline().export_video(source_path(media), output, settings, mode=mode,
                                         progress_callback=progress, cancel_event=event, preview=is_preview)
        # A successful return means publication already completed. A late Stop
        # must not hide a valid export that won the cancellation race.
        update_job(identifier, status='completed', progress=1, message='Export verified and ready',
                   output_url=f'/api/jobs/{identifier}/download', output_name=name, result=result)
    except Exception as error:
        update_job(identifier, status='cancelled' if event.is_set() else 'failed',
                   message='Render cancelled' if event.is_set() else 'Export failed', error=str(error))
    finally:
        if acquired:
            GPU_LOCK.release()


@app.post('/api/jobs')
def start_job(request: JobRequest):
    media = get_media(request.media_id)
    if request.mode not in ('match', 'hevc', 'prores', 'lossless'):
        raise HTTPException(422, 'Choose Match source, HEVC, ProRes or lossless master.')
    settings = validated_settings(request.settings)
    with STATE_LOCK:
        if sum(j['status'] in ('queued', 'running', 'cancelling') for j in JOBS.values()) >= 4:
            raise HTTPException(409, 'Four renders are already queued. Wait or cancel one.')
        identifier = uuid.uuid4().hex
        folder = STORE / 'jobs' / identifier
        folder.mkdir(parents=True)
        job = {'id': identifier, 'media_id': media['id'], 'status': 'queued', 'progress': 0,
               'message': 'Queued for the local GPU', 'output_url': None, 'output_name': None,
               'error': None, 'preview': request.preview, 'mode': request.mode, 'created_at': time.time()}
        JOBS[identifier] = job
        CANCEL[identifier] = threading.Event()
        save_json(folder / 'job.json', job)
        EXECUTOR.submit(execute_job, identifier, media, settings, request.mode, request.preview)
        return public_job(job)


@app.get('/api/jobs/{identifier}')
def job_status(identifier: str):
    with STATE_LOCK:
        if identifier not in JOBS:
            raise HTTPException(404, 'Render not found')
        return public_job(JOBS[identifier])


@app.post('/api/jobs/{identifier}/cancel')
def cancel_job(identifier: str):
    with STATE_LOCK:
        job = job_status(identifier)
        if job['status'] in ('queued', 'running', 'cancelling'):
            CANCEL[identifier].set()
            return update_job(identifier, status='cancelling', message='Stopping this render…')
        return job


@app.get('/api/jobs/{identifier}/download')
def download(identifier: str):
    job = job_status(identifier)
    if job['status'] != 'completed' or not job['output_name']:
        raise HTTPException(409, 'The verified export is not ready.')
    path = STORE / 'jobs' / identifier / job['output_name']
    return FileResponse(path, filename=job['output_name'], media_type=mimetypes.guess_type(path)[0] or 'application/octet-stream')


@app.post('/api/shutdown')
def shutdown():
    for event in CANCEL.values():
        event.set()
    if SERVER:
        SERVER.should_exit = True
    return {'stopping': True}


if (ROOT / 'app/dist').is_dir():
    app.mount('/', StaticFiles(directory=ROOT / 'app/dist', html=True), name='studio')
else:
    @app.get('/')
    def build_first():
        return {'message': 'Build the Studio interface with npm --prefix app run build, then restart.'}


if __name__ == '__main__':
    import uvicorn
    SERVER = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=8766, log_level='info'))
    SERVER.run()
