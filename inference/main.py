import os 
import sys
sys.path.insert(1, "/".join(os.getcwd().split("/")[:-1]))
import time
import cv2
import numpy as np
from threading import Thread
from queue import Queue
from models.model import Model
from training.data_augmentation import transform_normalize
import torch
from opcua import Client, ua
from omegaconf import OmegaConf


class ReolinkCamera:
    def __init__(self):

        self.cfg = OmegaConf.load("../cfg.yaml")
        
        self.cam_username = self.cfg.inference.camera.username
        self.cam_password = self.cfg.inference.camera.password
        self.cam_ip = self.cfg.inference.camera.ip
        self.cam_port = self.cfg.inference.camera.port
        
        self.model_weights = "./MainModel.pt"
        self.device = self.cfg.inference.model.device
        
        self.model = self.initialize_model()
        
        self.transform_normalize = transform_normalize.to(self.device)
        
        self.queues = [Queue(), Queue()]

        self.opcua_url = self.cfg.inference.opcua.url
        self.client = Client(self.opcua_url)

        while True:
            ret = self.connect_to_server()
            if ret:
                break

        self.node_lists = self.get_opcua_nodes_list()

        self.read_cam_left_thread = Thread(target=self.read_camera,
                                          args=(0,))
        self.read_cam_right_thread = Thread(target=self.read_camera,
                                          args=(1,))
        
        self.crop_size = self.cfg.inference.crop_size
        self.crop_coords = self.get_crop_coords()
        self.input_size = self.cfg.inference.model.input_size
        self.threshold = self.cfg.inference.model.threshold

    def connect_to_server(self):
        try:
            self.client.connect()
            print("Successfully connected to OPCUA server")
            return True
        except:
            print("THE SERVER IS UNREACHABLE")
            return False

    def get_opcua_nodes_list(self):
        
        node_list_left = [
            self.client.get_node('ns=3;s="Lights"."positionOccupied"[10]'),
            self.client.get_node('ns=3;s="Lights"."positionOccupied"[9]'),
            self.client.get_node('ns=3;s="Lights"."positionOccupied"[8]'),
            self.client.get_node('ns=3;s="Lights"."positionOccupied"[7]')
        ]
        node_list_right = [
            self.client.get_node('ns=3;s="Lights"."positionOccupied"[4]'),
            self.client.get_node('ns=3;s="Lights"."positionOccupied"[3]'),
            self.client.get_node('ns=3;s="Lights"."positionOccupied"[2]'),
            self.client.get_node('ns=3;s="Lights"."positionOccupied"[1]')
        ]
        nodes_lists = [node_list_left, node_list_right]

        return nodes_lists 

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

    def run(self):
        self.read_cam_left_thread.daemon = True
        self.read_cam_right_thread.daemon = True

        self.read_cam_left_thread.start()
        self.read_cam_right_thread.start()

        self.read_cam_left_thread.join()
        self.read_cam_right_thread.join()

    @staticmethod
    def get_ua_boolean_object(boolean_value):
        if type(boolean_value) is not bool:
            raise TypeError(f'Parameter boolean_value is not boolean (passed {type(boolean_value)} ).')
        ua_boolean = ua.DataValue(ua.Variant(boolean_value, ua.VariantType.Boolean))
        ua_boolean.ServerTimestamp = None
        ua_boolean.SourceTimestamp = None
        return ua_boolean

    def update_results(self, results, nodes):
        results_opcua_array = [self.get_ua_boolean_object(results[0]),
                               self.get_ua_boolean_object(results[1]),
                               self.get_ua_boolean_object(results[2]),
                               self.get_ua_boolean_object(results[3])]
        try:
            self.client.set_values(nodes, results_opcua_array)
            print("Results updated")
        except:
            print("Failed to update results")
            self.connect_to_server()

    def get_camera_url(self, index):
        url = f"tsp://{self.cam_username}:{self.cam_password}@{self.cam_ip}:{self.cam_port}/h264Preview_0{index + 1}_main"
        return url

    def read_camera(self, camera_index):
        camera_url = self.get_camera_url(camera_index)
        cap = cv2.VideoCapture(camera_url)
        frame_counter = 0
        since = time.time()

        while True:
            ret, frame = cap.read()
            if ret:
                predictions = self.get_prediction(frame, camera_index).tolist()
                self.update_results(predictions, self.node_lists[camera_index])
                frame_counter += 1
                fps_str = str(frame_counter / (time.time() - since))[:4]
                print("FPS camera {}".format(camera_index), fps_str, predictions)

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
    camera_object = ReolinkCamera()
    camera_object.run()
