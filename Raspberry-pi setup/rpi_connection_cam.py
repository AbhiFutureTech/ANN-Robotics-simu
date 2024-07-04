
__main-author__ = 'zhengwang'
__co-author__ = 'akshay'

import io
import socket
import struct
import time
import picamera


# create socket and bind host   192.168.43.166    192.168.137.1
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('10.42.0.1', 8000))
connection = client_socket.makefile('wb')
buffe=20
try:
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)      # pi camera resolution
        camera.framerate = 10               # 10 frames/sec
        time.sleep(2)                       # give 2 secs for camera to initilize
        start = time.time()
        stream = io.BytesIO()
        
        # send jpeg format video stream
        for foo in camera.capture_continuous(stream, 'jpeg', use_video_port = True):
            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()
            stream.seek(0)
            connection.write(stream.read())
            if time.time() - start > 600:
                break
            stream.seek(0)
            stream.truncate()
            
    connection.write(struct.pack('<L', 0))
finally:
    connection.close()
    client_socket.close()
