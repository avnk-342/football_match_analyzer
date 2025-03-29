from ultralytics import YOLO
import cv2
import supervision as sv
import pickle
import os
import numpy as np
import sys
sys.path.append('../')
from utils import get_centre_of_bbox, get_width_of_bbox

class Tracker:
    
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch

        return detections

    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = bbox[3]
        x_centre , _ = get_centre_of_bbox(bbox)
        width = get_width_of_bbox(bbox)

        cv2.ellipse(
            frame,
            center = (round(x_centre), round(y2)), 
            axes = (round(width), round(0.35*width)),
            angle=0.0,
            startAngle=-45, 
            endAngle=235, 
            color = color, 
            thickness = 2, 
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_centre - rectangle_width // 2
        x2_rect = x_centre + rectangle_width // 2
        y1_rect = y2 - rectangle_height // 2 + 15
        y2_rect = y2 + rectangle_height // 2 + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (round(x1_rect), round(y1_rect)),
                (round(x2_rect), round(y2_rect)),
                color,
                cv2.FILLED
            )

            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (round(x1_text), round(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            ) 

        return frame


    def get_object_tracks(self, frames, read_from_stub = False, stub_path = None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks


        detections = self.  detect_frames(frames)

        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_name_inverse = { value:key for key, value in class_names.items() }
            
            
            #converting detection to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = class_name_inverse["player"]
            
            #Track objects
            detection_with_tracker = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracker:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_name_inverse["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                
                if class_id == class_name_inverse["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == class_name_inverse["ball"]:
                    tracks["ball"][frame_num][1] = {'bbox': bbox}
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_triangle(self,frame,bbox,color):
        y = int(bbox[1])
        x,_ = get_centre_of_bbox(bbox)
        triangle_points = np.array([[x,y], [x-10, y-20], [x+10, y-20]], dtype=np.int32)
        cv2.drawContours(frame, [triangle_points], 0, color, -1)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame


    
    def draw_annotations(self, video_frame, tracks):
        output_video_frame = []
        for frame_num, frame in enumerate(video_frame):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            #drawing players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            #Drawing referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255))

            #draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))

            output_video_frame.append(frame)
        return output_video_frame