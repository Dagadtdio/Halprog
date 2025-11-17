import argparse
from collections import defaultdict, deque
import supervision as sv
import cv2
import numpy as np
from ultralytics import YOLO


SOURCE= np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source= source.astype(np.float32)
        target= target.astype(np.float32)
        self.m= cv2.getPerspectiveTransform(source,target)
    
    def transform_points(self, points: np.ndarray)->np.ndarray:
        reshaped_points = points.reshape(-1,1,2).astype(np.float32)
        transformed_points= cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1,2)

def parse_arguments() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Jármű gyorsaság számláló")
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="A videó helye",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Elkészített videó kimenete",
        type=str,
    )
    return parser.parse_args()


if __name__=="__main__":
    args=parse_arguments()

    video_info= sv.VideoInfo.from_video_path(args.source_video_path)
    model = YOLO("yolov8x.pt")
    CLASS_NAMES = model.names
    byte_track= sv.ByteTrack(frame_rate=video_info.fps)

    thickness= sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    corner_annotator = sv.BoxCornerAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.CLASS)
    label_annotator= sv.LabelAnnotator(text_scale=text_scale,text_thickness=thickness, color_lookup=sv.ColorLookup.CLASS)
    trace_annotator= sv.TraceAnnotator(thickness=thickness, trace_length=video_info.fps * 2,
                                       position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.CLASS)

    frame_generator =sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(SOURCE)

    view_transformer= ViewTransformer(source=SOURCE,target=TARGET)

    coordinates= defaultdict(lambda:deque(maxlen=video_info.fps))
    
    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            result= model(frame)[0]
            detections=sv.Detections.from_ultralytics(result)
            detections=detections[polygon_zone.trigger(detections)]
            detections= byte_track.update_with_detections(detections=detections)

            points= detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points= view_transformer.transform_points(points=points).astype(int)
            labels= []
            for tracker_id, class_id, [_, y] in zip(detections.tracker_id,
                                            detections.class_id,
                                            points):
                class_name = CLASS_NAMES[int(class_id)]

                coordinates[tracker_id].append(y)

                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id} {class_name}")
                else:
                    coordinates_start = coordinates[tracker_id][-1]
                    coordinates_end = coordinates[tracker_id][0]
                    distance = abs(coordinates_start - coordinates_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6

                    labels.append(f"#{tracker_id} {class_name} {int(speed)} km/h")


            

            annotated_frame= frame.copy()
            annotated_frame= trace_annotator.annotate(scene=annotated_frame, 
                                                    detections=detections)
            annotated_frame= sv.draw_polygon(annotated_frame,
                                            polygon=SOURCE,
                                            color= sv.Color.RED)
            annotated_frame= corner_annotator.annotate(scene=annotated_frame,
                                                    detections=detections)
            annotated_frame= label_annotator.annotate(scene=annotated_frame,
                                                    detections=detections,
                                                    labels=labels)
            sink.write_frame(annotated_frame)
            cv2.imshow("frame", annotated_frame)
            if cv2.waitKey(1) == ord("q"):
                break
    cv2.destroyAllWindows()