#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import traceback
import socket
import json
import struct
from scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)

def send_json_data(conn, data):
    # Serialize the list of strings to JSON
    serialized_data = json.dumps(data)
    # Convert the serialized data to bytes
    bytes_data = serialized_data.encode('utf-8')
    # Send the length of the serialized data first
    conn.sendall(struct.pack('I', len(bytes_data)))
    # Send the actual serialized data
    conn.sendall(bytes_data)

def try_connect(render_items):
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        # print(f"\nConnected by {addr}")
        conn.settimeout(0.0)
        send_json_data(conn, render_items)
    except Exception as inst:
        pass
        # raise inst
            
def _recv_exact(n: int):
    """Receive exactly n bytes or return None if no data (non-blocking)."""
    global conn
    data = b""
    while len(data) < n:
        try:
            chunk = conn.recv(n - len(data))
        except (BlockingIOError, socket.timeout):
            return None  # no data available right now
        if chunk == b"":
            # disconnected
            raise ConnectionError("GUI disconnected")
        data += chunk
    return data

def read():
    global conn
    # try read header
    hdr = _recv_exact(4)
    if hdr is None:
        return None
    messageLength = int.from_bytes(hdr, 'little')
    body = _recv_exact(messageLength)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))

def send(message_bytes, verify, metrics):
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))
    send_json_data(conn, metrics)

def receive():
    global conn
    try:
        message = read()
        if message is None:
            # No GUI message available (non-blocking)
            return None, True, True, 1.0, 0

        width = message["resolution_x"]
        height = message["resolution_y"]

        if width != 0 and height != 0:
            do_training = bool(message.get("train", True))
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            keep_alive = bool(message.get("keep_alive", True))
            scaling_modifier = message.get("scaling_modifier", 1.0)

            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]

            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]

            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar,
                                 world_view_transform, full_proj_transform)
            render_mode = message.get("render_mode", 0)
            return custom_cam, do_training, keep_alive, scaling_modifier, render_mode

        return None, True, True, 1.0, 0

    except Exception:
        # Any error -> drop connection so training continues
        conn = None
        return None, True, True, 1.0, 0
