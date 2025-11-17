from PyQt5.QtCore import QThread
import subprocess
import numpy as np
import cv2
from ..models import CameraConfig, mainConfig
from queue import Queue


class ThreadStreaming(QThread):
    def __init__(self, parent=None, config_camera: CameraConfig = None, stream_queue: Queue = None):
        super().__init__(parent=parent)
        self.__thread_active = False
        self.stream_queue = stream_queue
        self.config_camera = config_camera

        stream_url = f"{mainConfig.STREAM_URL}_{config_camera.compId}/{config_camera.code}"
        
        # LOW LATENCY FFMPEG for real-time streaming
        self.args = (
            f"ffmpeg -re -r 15 -f rawvideo -vcodec rawvideo -pix_fmt "
            f"bgr24 -s {mainConfig.STREAM_SIZE[0]}x{mainConfig.STREAM_SIZE[1]} -i pipe:0 "
            f"-pix_fmt yuv420p -c:v libx264 -preset ultrafast -tune zerolatency "
            f"-b:v 800k -maxrate 800k -bufsize 400k "
            f"-g 30 -keyint_min 30 -sc_threshold 0 "
            f"-flush_packets 1 "
            f"-f flv {stream_url} -loglevel error"
        ).split()
        print(f"{self.config_camera.name} started stream at ", stream_url)
        self.recognize_success_dict = []

    def on_recognize_success(self, recognize_success_dict):
        self.recognize_success_dict = recognize_success_dict

    @property
    def is_active(self):
        return self.__thread_active

    def run(self):
        print("Thread Streaming Started")
        self.__thread_active = True
        self.ffmpeg_process = subprocess.Popen(
            self.args, stdin=subprocess.PIPE)

        while self.__thread_active:
            if self.stream_queue.empty():
                self.msleep(5)
                continue

            frame, id_dict, is_tracking = self.stream_queue.get()
            while not self.stream_queue.empty():
                try:
                    frame, id_dict, is_tracking = self.stream_queue.get_nowait()
                except:
                    break
            
            tracking_text = "Tracking" if is_tracking else "Not Tracking"
            tracking_text_color = (0, 255, 0) if is_tracking else (0, 0, 255)
            frame_copy: np.ndarray = frame.copy()
            color = (0, 0, 255)
            for key, bbox in id_dict.items():
                x1, y1, x2, y2, score, cls = bbox
                value = "Unknown"
                if key in self.recognize_success_dict:
                    color = (0, 255, 0)
                    value = self.recognize_success_dict[key]
                else:
                    if cls == 0:
                        color = (255, 0, 0)
                    else:
                        color = (0, 0, 255)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_copy, str(value), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame_copy, tracking_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, tracking_text_color, 2)
            if frame_copy.shape[:2] != mainConfig.STREAM_SIZE[::-1]:
                frame_copy = cv2.resize(
                    frame_copy, tuple(mainConfig.STREAM_SIZE))
            try:
                self.ffmpeg_process.stdin.write(frame_copy.tobytes())
            except Exception as e:
                print(f"Error Stream: {self.config_camera.link}", e)
                self.ffmpeg_process = subprocess.Popen(
                    self.args, stdin=subprocess.PIPE)

            self.msleep(10)

        try:
            self.ffmpeg_process.kill()
            self.ffmpeg_process.wait()
        except:
            print("Unexpectly error occurs when streaming process's closed ")

    def stop(self):
        self.__thread_active = False
        try:
            self.ffmpeg_process.kill()
            self.ffmpeg_process.wait()
        except:
            print("Unexpectly error occurs when streaming process's closed ")
        print("ThreadStreaming stopped")
