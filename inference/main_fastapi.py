import os 
import sys
sys.path.insert(1, "/".join(os.getcwd().split("/")[:-1]))
import time
import cv2
import numpy as np
from threading import Thread, Lock
from queue import Queue
from models.model import Model
from training.data_augmentation import transform_normalize
import torch
# from opcua import Client, ua
from omegaconf import OmegaConf

from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
import uvicorn


class RobobarGlassRecognition:
    def __init__(self):

        self.cfg = OmegaConf.load("../cfg.yaml")
        
        self.cam_username = self.cfg.inference.camera.username
        self.cam_password = self.cfg.inference.camera.password
        self.cam_ip = self.cfg.inference.camera.ip
        self.cam_port = self.cfg.inference.camera.port
        
        self.model_weights = "./main_model.pt"
        self.device = self.cfg.inference.model.device
        
        self.model = self.initialize_model()
        
        self.transform_normalize = transform_normalize.to(self.device)
        
        self.queues = [Queue(), Queue()]
        self.lock = [Lock(), Lock()]

        self.latest_predictions = self.init_predictions()
        
        self.read_cam_left_thread = Thread(target=self.get_and_process,
                                          args=(0,))
        self.read_cam_right_thread = Thread(target=self.get_and_process,
                                          args=(1,))

        
        self.crop_size = self.cfg.inference.model.crop_size
        self.crop_coords = self.get_crop_coords()
        self.input_size = self.cfg.inference.model.input_size
        self.threshold = self.cfg.inference.model.threshold

        self.fastapi_host = self.cfg.inference.fastapi.host
        self.fastapi_port = self.cfg.inference.fastapi.port


        self.app = FastAPI()
        self.setup_api_routes()


    def run_fastapi_app(self):
        print("🚀 Starting FastAPI server. Go to http://127.0.0.1:8000/results")
        uvicorn.run(self.app, host=self.fastapi_host, port=self.fastapi_port)


    def setup_api_routes(self):
        @self.app.get("/")
        def read_root():
            return {"message": "Functional Update API is running. Call the `update_results` function to change the state."}

        @self.app.get("/results", response_model=Dict[int, List[bool]])
        def get_results_by_camera():
            return self.latest_predictions
    
    def init_predictions(self):
        predictions = {0: [None, None, None, None],
                        1: [None, None, None, None]}
        
        return predictions
                      

    def update_predictions(self, camera_index, results):
        with self.lock[camera_index]:
            self.latest_predictions[camera_index] = results

    def get_crop_coords(self):
        crop_coords = {"0": (1325 + self.crop_size, 1220),
                               "1": (1460 + self.crop_size, 1140),
                               "2": (1625 + self.crop_size, 1070),
                               "3": (1830 + self.crop_size, 1010),

                               "4": (2750, 980),
                               "5": (2955, 1030),
                               "6": (3155, 1075),
                               "7": (3330, 1150),
                               }

        return crop_coords

    def initialize_model(self):
        model = Model()
        model.load_state_dict(torch.load(self.model_weights, map_location="cpu"))
        model.eval()
        model.to(self.device)

        return model

    def run_prediction_app(self):
        self.read_cam_left_thread.daemon = True
        self.read_cam_right_thread.daemon = True

        self.read_cam_left_thread.start()
        self.read_cam_right_thread.start()



    def get_camera_url(self, cam_idx):
        url = f"tsp://{self.cam_username}:{self.cam_password}@{self.cam_ip}:{self.cam_port}/h264Preview_0{cam_idx + 1}_main"
        return url

    def get_and_process(self, camera_index):
        camera_url = self.get_camera_url(camera_index)
        cap = cv2.VideoCapture(camera_url)

        while True:
            ret, frame = cap.read()
            if ret:
                predictions = self.get_prediction(frame, camera_index).tolist()
                self.update_predictions(camera_index, predictions)

                time.sleep(0.05)


    # def get_and_process(self, camera_index):
    #     while True:
    #         predictions = np.random.choice([True, False], size=4).tolist()  # Simulated predictions
    #         self.update_predictions(camera_index, predictions)

    #         time.sleep(0.05)

    def extract_squares(self, frame, camera_index):
        crops = []
        for position, coords in self.crop_coords[camera_index].items():
            x, y = coords
            if int(position) > 3:
                crop = frame[y:y + self.crop_size, x:x + self.crop_size, :]
            else:
                crop = frame[y:y + self.crop_size, x - self.crop_size:x, :]

            crop = cv2.resize(crop, (self.input_size, self.input_size))
            crop = self.transform_to_tensor(crop)

            crops.append(crop)

        crops = np.stack(crops, axis=0)
        return crops

    def get_prediction(self, frame, camera_index):
        crops = self.extract_squares(frame, camera_index)
        crops = torch.from_numpy(crops).to(self.device).float() / 255.0
        crops = transform_normalize(crops)

        logits = self.model(crops).reshape(-1) 

        probs = torch.nn.functional.sigmoid(logits)
        binary_preds = probs > self.threshold
        return binary_preds


if __name__ == "__main__":
    robobar_glass_recognition = RobobarGlassRecognition()
    print(f"Starting Robobar Glass Recognition")
    robobar_glass_recognition.run_prediction_app()
    print(f"Starting FastAPI server")
    robobar_glass_recognition.run_fastapi_app()
