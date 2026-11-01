# Anamorphic Studio client

Local cinema grading workspace for the FastAPI/CUDA service in `tools/anamorphic-studio`. React 19.3.1, TypeScript 7.0.2 and Vite 8.3.5; dependencies are pinned by `package-lock.json`. System fonts and local Lucide SVG icons require no CDN or external assets.

Run `npm ci`, then `npm run build` to produce `app/dist`, which the local backend serves at `http://127.0.0.1:8766`. `npm run lint` checks the client. For development, run `npm run dev`; Vite binds to loopback and proxies `/api` to the local backend. `VITE_API_BASE` optionally selects another locally operated backend. No Vercel configuration or services are included.

The client supports file import/drop, source metadata, all 19 engine settings, four lens profiles, settings JSON save/load, debounced exact-frame GPU previews, a draggable comparison, first-three-second rendered previews, full exports, cancellation and downloads. Look controls lock while a render runs; changing a look invalidates its previous rendered clip. Each successful export displays the backend's actual codec, pixel format, quality, audio outcome and warnings.

Match source retains source dimensions and frame timing. The backend validates supported codec, pixel and audio combinations. A lossless codec on source YUV samples does not make RGB conversion mathematically lossless; the separate lossless RGB master preserves processed RGB frames and can produce much larger files. These effects do not add camera capture resolution. Focus falloff uses an artistic focus band rather than estimated scene depth.

Uploads and rendered media travel directly between this browser and the local backend. No media bytes pass through Vercel. The API requires `X-Studio-Client: anamorphic-studio` on POST requests, including imports. Framework references used for this implementation: [React effects](https://react.dev/reference/react/useEffect), [Vite server options](https://vite.dev/config/server-options) and [Vite static builds](https://vite.dev/guide/static-deploy.html).
