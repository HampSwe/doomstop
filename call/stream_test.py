"""Video both ways between the presenter Mac and the audience PC, measured.

This checks the demo call's transport before anything else is built on it.
Each side sends JPEG frames over one websocket. The other side decodes every
frame and acknowledges it, and both sides print how many frames per second
arrive, the round-trip time and the bandwidth. It ends with PASS or FAIL.

Presenter Mac (start it first):
    .venv/bin/python call/stream_test.py serve
Audience PC (the Mac's address on your own network):
    .venv\\Scripts\\python.exe call\\stream_test.py connect 10.42.0.1

Options (either side): --camera N sends a real camera instead of the moving
test picture, --width 640 --fps 10 --quality 70 set the stream, --seconds 60
sets how long, and --show opens a window with what arrives.

Frames are never queued: with MAX_IN_FLIGHT frames still unacknowledged, the
next one is dropped and counted, so a slow link shows up as dropped frames
instead of growing delay. The client reconnects by itself if the link drops.
Needs only aiohttp, numpy and opencv on both machines.
"""

import argparse
import asyncio
import json
import socket
import struct
import sys
import time

import cv2
import numpy as np
from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web

PORT = 8770
HEADER = struct.Struct("<I")       # frame number, then the JPEG bytes
MAX_IN_FLIGHT = 2
PASS_FPS_SHARE = 0.9               # at least 90% of the frames asked for arrive
PASS_RTT_P95_S = 0.25              # and 95% of them come back within 250 ms


class Source:
    """Frames to send: a real camera, or a moving, textured test picture of realistic JPEG size."""

    def __init__(self, camera, width):
        self.width, self.n, self.cap = width, 0, None
        if camera is not None:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(camera, backend)
            if not self.cap.isOpened():
                sys.exit(f"camera {camera} did not open")
            return
        h = width * 9 // 16
        y, x = np.mgrid[0:h, 0:width]
        rng = np.random.default_rng(0)
        base = np.dstack([(x * 255 // width), (y * 255 // h), np.full_like(x, 128)]).astype(np.int16)
        self.picture = np.clip(base + rng.integers(-25, 25, base.shape), 0, 255).astype(np.uint8)

    def frame(self):
        self.n += 1
        if self.cap is not None:
            ok, f = self.cap.read()
            if not ok:
                return None
            return cv2.resize(f, (self.width, f.shape[0] * self.width // f.shape[1]))
        f = np.roll(self.picture, self.n * 7, axis=1).copy()
        cv2.putText(f, f"{socket.gethostname()}  frame {self.n}  {time.strftime('%H:%M:%S')}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, self.width / 900, (255, 255, 255), 2)
        return f


class Stats:
    def __init__(self):
        self.seq = self.sent = self.acked = self.dropped = self.received = 0
        self.bytes_out = self.bytes_in = 0
        self.rtts, self.reconnects, self.connected_s = [], 0, 0.0
        self.last_image = None
        self.done = False


def pct(values, q):
    return float(np.percentile(values, q)) * 1000 if values else float("nan")


async def peer(ws, source, args, st, deadline):
    """Send and receive over one open websocket until the deadline, the far side's 'bye' or a drop."""
    in_flight = {}
    started = time.perf_counter()

    async def send():
        period, due = 1 / args.fps, time.perf_counter()
        while time.time() < deadline and not ws.closed and not st.done:
            due += period
            if len(in_flight) >= MAX_IN_FLIGHT:
                st.dropped += 1                                  # the far side is behind: skip, don't queue
            else:
                f = await asyncio.to_thread(source.frame)
                if f is not None:
                    ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                    st.seq += 1
                    in_flight[st.seq] = time.perf_counter()
                    await ws.send_bytes(HEADER.pack(st.seq) + jpg.tobytes())
                    st.sent += 1
                    st.bytes_out += len(jpg)
            await asyncio.sleep(max(0.0, due - time.perf_counter()))

    async def receive():
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                (seq,) = HEADER.unpack_from(msg.data)
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8, offset=HEADER.size), cv2.IMREAD_COLOR)
                await ws.send_str(json.dumps({"ack": seq}))
                if img is not None:
                    st.received += 1
                    st.bytes_in += len(msg.data)
                    st.last_image = img
            elif msg.type == WSMsgType.TEXT:
                m = json.loads(msg.data)
                if m.get("bye"):
                    st.done = True
                    return
                t = in_flight.pop(m.get("ack"), None)
                if t is not None:
                    st.acked += 1
                    st.rtts.append(time.perf_counter() - t)

    sender = asyncio.create_task(send())
    try:
        await asyncio.wait([sender, asyncio.create_task(receive())], return_when=asyncio.FIRST_COMPLETED)
    finally:
        sender.cancel()
        st.connected_s += time.perf_counter() - started


async def report(st, args, label):
    """A line every 2 s, and a window with the latest frame if --show."""
    prev, last_print = (0, 0, 0, 0, 0), time.perf_counter()
    while not st.done:
        await asyncio.sleep(0.03 if args.show else 0.5)
        if args.show and st.last_image is not None:
            cv2.imshow(f"stream_test {label}: what arrives", st.last_image)
            cv2.waitKey(1)
        now = time.perf_counter()
        if now - last_print >= 2:
            dt, last_print = now - last_print, now
            cur = (st.acked, st.received, st.bytes_out, st.bytes_in, len(st.rtts))
            d = [c - p for c, p in zip(cur, prev)]
            prev = cur
            recent = st.rtts[-d[4]:] if d[4] else []
            print(f"  out {d[0] / dt:5.1f} fps {d[2] / dt / 1e6:5.2f} MB/s | in {d[1] / dt:5.1f} fps "
                  f"{d[3] / dt / 1e6:5.2f} MB/s | round trip p50 {pct(recent, 50):4.0f} ms "
                  f"p95 {pct(recent, 95):4.0f} ms | dropped {st.dropped} | reconnects {st.reconnects}",
                  flush=True)


def summary(st, args):
    secs = max(st.connected_s, 1e-9)
    out_fps, in_fps = st.acked / secs, st.received / secs
    ok_fps = out_fps >= PASS_FPS_SHARE * args.fps
    ok_rtt = st.rtts and pct(st.rtts, 95) / 1000 <= PASS_RTT_P95_S
    print(f"\nconnected {secs:.0f} s, {st.reconnects} reconnects")
    print(f"sent    {st.sent} frames, {out_fps:.1f} fps delivered (asked {args.fps}), "
          f"{st.bytes_out / max(st.sent, 1) / 1000:.0f} KB each, dropped {st.dropped}")
    print(f"received {st.received} frames, {in_fps:.1f} fps")
    print(f"round trip (send, decode, ack): p50 {pct(st.rtts, 50):.0f} ms, p95 {pct(st.rtts, 95):.0f} ms, "
          f"max {pct(st.rtts, 100):.0f} ms")
    print("RESULT:", "PASS" if ok_fps and ok_rtt else
          f"FAIL ({'too few frames delivered' if not ok_fps else 'round trip too slow'})")


async def serve(args):
    st, source = Stats(), Source(args.camera, args.width)
    busy = asyncio.Lock()

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=2, max_msg_size=16 * 2**20)
        await ws.prepare(request)
        if busy.locked():
            await ws.close(message=b"another peer is connected")
            return ws
        async with busy:
            print(f"peer connected from {request.remote}", flush=True)
            await peer(ws, source, args, st, time.time() + 24 * 3600)
            print("peer gone" if not st.done else "peer finished", flush=True)
        return ws

    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", args.port).start()
    print(f"listening on port {args.port}. On the other machine:\n"
          f"  python call/stream_test.py connect <this machine's address>\n"
          f"  (this Mac's Wi-Fi address: `ipconfig getifaddr en0`; name: {socket.gethostname()})", flush=True)
    try:
        await report(st, args, "serve")
    finally:
        await runner.cleanup()
    summary(st, args)


async def connect(args):
    st, source = Stats(), Source(args.camera, args.width)
    url = f"http://{args.host}:{args.port}/ws"
    deadline = time.time() + args.seconds
    reporter = asyncio.create_task(report(st, args, "connect"))
    backoff, first = 0.5, True
    async with ClientSession(timeout=ClientTimeout(total=None, connect=3)) as http:
        while time.time() < deadline:
            try:
                async with http.ws_connect(url, heartbeat=2, max_msg_size=16 * 2**20) as ws:
                    if not first:
                        st.reconnects += 1
                    print(f"connected to {url}", flush=True)
                    first, backoff = False, 0.5
                    await peer(ws, source, args, st, deadline)
                    if time.time() >= deadline:
                        await ws.send_str(json.dumps({"bye": True}))
                        break
                    print("link dropped, reconnecting ...", flush=True)
            except (ClientError, OSError, asyncio.TimeoutError) as e:
                print(f"cannot reach {url} ({type(e).__name__}), retrying in {backoff:.1f} s", flush=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 2.0)
    st.done = True
    await reporter
    summary(st, args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["serve", "connect"])
    p.add_argument("host", nargs="?", help="connect: the presenter Mac's address")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--camera", type=int, help="camera index to send instead of the test picture")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--quality", type=int, default=70)
    p.add_argument("--seconds", type=float, default=60, help="connect: how long to run")
    p.add_argument("--show", action="store_true", help="open a window with what arrives")
    args = p.parse_args()
    if args.mode == "connect" and not args.host:
        p.error("connect needs the presenter Mac's address")
    try:
        asyncio.run(serve(args) if args.mode == "serve" else connect(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
