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
        print("[network_gui] CONNECTED:", addr, flush=True)
        print("[network_gui] render_items:", render_items, flush=True)

        conn.settimeout(None)

        # IMPORTANT: disable this for compatibility
        send_json_data(conn, render_items)

    except BlockingIOError:
        return
    except Exception as inst:
        print("[network_gui.try_connect] ERROR:", repr(inst), flush=True)
        traceback.print_exc()
            
def read():
    global conn

    print("[network_gui.read] waiting header...", flush=True)

    messageLength = conn.recv(4)

    print("[network_gui.read] raw header bytes:", messageLength, flush=True)

    messageLength = int.from_bytes(messageLength, 'little')

    print("[network_gui.read] messageLength:", messageLength, flush=True)

    message = conn.recv(messageLength)

    print("[network_gui.read] received body bytes:", len(message), flush=True)
    print("[network_gui.read] body text:", message[:500], flush=True)

    parsed = json.loads(message.decode("utf-8"))

    print("[network_gui.read] parsed keys:", parsed.keys(), flush=True)

    return parsed

def send(message_bytes, verify, metrics):
    global conn

    print("[network_gui.send] ENTER", flush=True)

    if message_bytes != None:
        print("[network_gui.send] image bytes:", len(message_bytes), flush=True)
        conn.sendall(message_bytes)
        print("[network_gui.send] image sent", flush=True)
    else:
        print("[network_gui.send] image is NONE", flush=True)

    print("[network_gui.send] verify:", verify, flush=True)

    conn.sendall(len(verify).to_bytes(4, 'little'))
    print("[network_gui.send] verify len sent", flush=True)

    conn.sendall(bytes(verify, 'ascii'))
    print("[network_gui.send] verify string sent", flush=True)

    print("[network_gui.send] metrics:", metrics, flush=True)

    send_json_data(conn, metrics)

    print("[network_gui.send] metrics sent", flush=True)

def receive():
    message = read()
    print("[network_gui.receive] FULL MESSAGE:", message, flush=True)
    width = message["resolution_x"]
    height = message["resolution_y"]

    print("[network_gui.receive] resolution:", width, height, flush=True)
    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
            render_mode = message["render_mode"]
            print("[network_gui.receive] BUILDING CAMERA", flush=True)

            print("[network_gui.receive] fovx/fovy:", fovx, fovy, flush=True)
            print("[network_gui.receive] znear/zfar:", znear, zfar, flush=True)
            print("[network_gui.receive] render_mode:", render_mode, flush=True)
        except Exception as e:
            print("")
            traceback.print_exc()
            # raise e
        return custom_cam, do_training, keep_alive, scaling_modifier, render_mode
    else:
        return None, None, None, None, None