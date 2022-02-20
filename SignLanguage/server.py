import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from av import VideoFrame
from flask import Flask

import numpy as np
import onnxruntime as ort

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

gameData = {'letter': ''}


audio_letters = {
    'A': 'A_Guitar.wav',
    'B': 'B_Guitar.wav',
    'C': 'C_Guitar.wav',
    'D': 'D_Guitar.wav',
    'E': 'Em_Guitar.wav',
    'F': 'F_Guitar.wav'
}


songs = {
    1: 'assets/guitar/chord (1).wav',
    2: 'assets/guitar/chord (2).wav',
    3: 'assets/guitar/chord (3).wav',
    4: 'assets/guitar/chord (4).wav',
    5: 'assets/guitar/chord (5).wav',
    6: 'assets/guitar/chord (6).wav'
}


class AudioTrack(MediaStreamTrack):

    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.tracks = []
        self.timer = 0

    async def recv(self):
        frame = await self.track.recv()       # preprocess data
        if self.timer == 48:
            # print("SENDING AUDIO")
            player = MediaPlayer(os.path.join(
                ROOT, "assets/guitar/chord (1).wav"))
            self.timer = 0
            return player.audio
        else:
            self.timer = self.timer + 1
            return frame


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

        # constants
        self.index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
        self.mean = 0.485 * 255.
        self.std = 0.229 * 255.
        self.timer = 0

        # create runnable session with exported model
        self.ort_session = ort.InferenceSession("signlanguage.onnx")

    async def recv(self):
        frame = await self.track.recv()       # preprocess data
        width = 700
        height = 480
        img = cv2.resize(frame.to_ndarray(format="bgr24"), (width, height))

        if self.timer == 8:
            cropImg = img[20:250, 20:250]
            img2 = cv2.cvtColor(cropImg, cv2.COLOR_RGB2GRAY)

            x = cv2.resize(img2, (28, 28))
            x = (x - self.mean) / self.std
            x = x.reshape(1, 1, 28, 28).astype(np.float32)
            y = self.ort_session.run(None, {'input': x})[0]

            index = np.argmax(y, axis=1)
            letter = self.index_to_letter[int(index)]
            gameData['letter'] = letter
            # print(gameData['audio'])
            # channel.send(gameData)
            print(letter)

            self.timer = 0
        else:
            self.timer = self.timer + 1

        cv2.putText(img, gameData['letter'], (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        img = cv2.rectangle(img, (20, 20), (250, 250), (0, 255, 0), 3)

        # rebuild a VideoFrame, preserving timing information

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        frame = new_frame

        return frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    if not int(params["song"]) == 0:
        song = songs[int(params["song"])]

        # # prepare local media
        # # TODO make this the song/sound we want to play
        player = MediaPlayer(os.path.join(ROOT, song))
        if args.record_to:
            recorder = MediaRecorder(args.record_to)
        else:
            recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):  # TODO make this return the game data
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + str(gameData))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)
        if track.kind == "audio":
            # pc.addTrack(AudioTrack(relay.subscribe(track)))
            # audio = AudioTrack(relay.subscribe(track))
            # pc.addTrack(track)

            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"]
                )
            )
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    # app.router.add_get("/", index)
    # app.router.add_get("/client.js", javascript)

    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
