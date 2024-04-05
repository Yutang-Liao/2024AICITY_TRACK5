
from ultralytics import YOLO
import cv2 
import torch
import torchvision
import numpy as np
output_labels =  ['motorbike'
,'DHelmet'
,'DNoHelmet'
, 'P1Helmet'
, 'P1NoHelmet'
, 'P2Helmet'
, 'P2NoHelmet'
, 'P0Helmet'
, 'P0NoHelmet'] 



def load_models(yolo_model_path_640,yolo_model_path_704,yolo_model_path_768,yolo_model_path_832,yolo_model_path_896,yolo_model_path_960,yolo_model_path_1024):
     
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_640 = YOLO(yolo_model_path_640) 
    model_704 = YOLO(yolo_model_path_704) 
    model_768 = YOLO(yolo_model_path_768) 
    model_832 = YOLO(yolo_model_path_832) 
    model_896 = YOLO(yolo_model_path_896) 
    model_960 = YOLO(yolo_model_path_960) 
    model_1024 = YOLO(yolo_model_path_1024) 

    print("YOLO model loaded!")


    return model_640,model_704,model_768,model_832,model_896,model_960,model_1024

def append_to_file(file_path, video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id, confidence):
    # 使用追加模式打開文件
    with open(file_path, 'a') as file:
        # 格式化字符串以匹配指定的格式
        data_line = f"{video_id},{frame},{bb_left},{bb_top},{bb_width},{bb_height},{class_id},{confidence}\n"
        # 將格式化好的字符串寫入文件
        file.write(data_line)


def is_within_threshold(center1, center2, threshold=10):
    """Check if the centers are within the threshold."""
    x1, y1 = center1
    x2, y2 = center2
    return abs(x1 - x2) <= threshold and abs(y1 - y2) <= threshold

def merge_multiple_model_outputs(*lists):
    combined = [item for lst in lists for item in lst]
    result = []  # This will store the final merged output

    while combined:
        current = combined.pop(0)
        i = 0
        while i < len(combined):
            # Compare centers and classes
            if current[1] == combined[i][1] and is_within_threshold(current[0][:2], combined[i][0][:2]):
                # If the same class and centers are within threshold, keep the one with higher confidence
                if current[2] < combined[i][2]:
                    current = combined[i]  # Replace with the item with higher confidence
                combined.pop(i)  # Remove the item with lower confidence or the duplicate
            else:
                i += 1
        result.append(current)

    return result

def main():
    import sys
    from ultralytics import YOLO
    import cv2 
    import torch
    import torchvision
    import numpy as np     # input_video_path = '/home/yutang/AICITY/aicity2024_track5_train/aicity2024_track5_train/yolov8/test_videos/videos/085.mp4'
    # output_video_path = 'output_video_085_bottrack.avi'
    number = sys.argv[1]
    format_number = number.zfill(3)
    input_video_path = f"./test_videos/{format_number}.mp4"
    output_video_path = f"./output_test_videos/{format_number}.mp4"
    file_path = './output_1024.txt'
    print(f"Processing video {format_number}.mp4")
    video_id = number ## need to change 
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


    frame_count = 0

    yolo_model_path_640 = 'yolov8x_640.pt'
    yolo_model_path_704 = 'yolov8m_704.pt'
    yolo_model_path_768 = 'yolov8x_768.pt'
    yolo_model_path_832 = "yolov8x_832.pt"
    yolo_model_path_896 = "yolov8x_896.pt"
    yolo_model_path_960= "yolov8x_960.pt"
    yolo_model_path_1024= "yolov8x_1024.pt"
    ## call function to load model 
    model_640,model_704,model_768,model_832,model_896,model_960,model_1024 = load_models(yolo_model_path_640,yolo_model_path_704,yolo_model_path_768,yolo_model_path_832,yolo_model_path_896,yolo_model_path_960,yolo_model_path_1024)

    while cap.isOpened():
        ret, frame = cap.read()
        frame_count +=1 
        if not ret or frame is None:
            print("Failed to read the frame or video has ended.")
            break

        model_640_results = model_640(frame  , conf = 0.1 , iou=0.7) #0.3
        model_704_results = model_704(frame  , conf = 0.1 , iou=0.7) #0.3
        model_768_results = model_768(frame  , conf = 0.1, iou=0.7) #0.3
        model_832_results = model_832(frame  , conf = 0.1 , iou=0.7) #0.3
        model_896_results = model_896(frame  , conf = 0.1 , iou=0.7) #0.3
        model_960_results = model_960(frame  , conf = 0.1 , iou=0.7)
        model_1024_results = model_1024(frame  , conf = 0.1 , iou=0.7)

        model_576_list = list()
        model_640_list = list()
        model_704_list = list()
        model_768_list = list()
        model_832_list = list()
        model_896_list = list()
        model_960_list = list()
        model_1024_list = list()

        for result in model_640_results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            ## Store Confidence
            for idx ,xywh in enumerate(boxes.xywh):
                conf = round(boxes.conf[idx].item(), 2)
                cls = int(boxes.cls[idx].item())
                model_640_list.append([xywh,cls,conf])

        for result in model_704_results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            ## Store Confidence
            for idx ,xywh in enumerate(boxes.xywh):
                conf = round(boxes.conf[idx].item(), 2)
                cls = int(boxes.cls[idx].item())
                model_704_list.append([xywh,cls,conf])

        for result in model_768_results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            ## Store Confidence
            for idx ,xywh in enumerate(boxes.xywh):
                conf = round(boxes.conf[idx].item(), 2)
                cls = int(boxes.cls[idx].item())
                model_768_list.append([xywh,cls,conf])
        for result in model_832_results:

            boxes = result.boxes  # Boxes object for bounding box outputs
            ## Store Confidence
            for idx ,xywh in enumerate(boxes.xywh):
                conf = round(boxes.conf[idx].item(), 2)
                model_832_list.append([xywh,cls,conf])

        for result in model_896_results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            ## Store Confidence
            for idx ,xywh in enumerate(boxes.xywh):
                conf = round(boxes.conf[idx].item(), 2)
                cls = int(boxes.cls[idx].item())
                model_896_list.append([xywh,cls,conf])

        for result in model_960_results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            ## Store Confidence
            for idx ,xywh in enumerate(boxes.xywh):
                conf = round(boxes.conf[idx].item(), 2)
                cls = int(boxes.cls[idx].item())
                model_960_list.append([xywh,cls,conf])

        for result in model_1024_results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            ## Store Confidence
            for idx ,xywh in enumerate(boxes.xywh):
                conf = round(boxes.conf[idx].item(), 2)
                cls = int(boxes.cls[idx].item())
                model_1024_list.append([xywh,cls,conf])

        merged_outputs = merge_multiple_model_outputs(model_576_list,model_640_list,model_704_list,model_768_list,model_832_list,model_896_list,model_960_list,model_1024_list)
                
        for xywh,cls,conf in merged_outputs:
            append_to_file(file_path, video_id, frame_count, int(xywh[0]-xywh[2]/2), int(xywh[1] - xywh[3]/2), int(xywh[2]), int(xywh[3]), cls+1, conf)
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()                


if __name__ == "__main__":
    main()

            
