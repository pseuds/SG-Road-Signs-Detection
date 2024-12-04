import sys
import numpy as np
import cv2 
import argparse
from ultralytics import YOLO
from pp import preprocess_img, resize_image

"""
Idea: see if any new prediction is similar to those found. If so, assign same id and 'track' it.
"""

def parseargs():
    parser = argparse.ArgumentParser(description='Meanshift Tracking')
    parser.add_argument('-v', '--video', type=str, required=True, help='Path to video file.')
    args = parser.parse_args()
    return args

def draw_bbox(img:np.ndarray, cls:int, bbox:tuple, id:int):
    start_point = (bbox[0]-bbox[2]//2, bbox[1]-bbox[3]//2)
    end_point = (bbox[0]+bbox[2]//2, bbox[1]+bbox[3]//2)

    cv2.rectangle(img, start_point, end_point, color=(255,255,0), thickness=1)
    label = f"cls:{cls}, id:{id}"
    cv2.putText(
        img,
        label,
        (int(start_point[0]), int(start_point[1]) - 10),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.6,
        color = (255, 255, 125),
        thickness=2
    )

def get_best_matching_obj(current_obj, frame, past_objs, exclude_ids=[]) -> int | None:
    """
    Returns the index of the past object that best matches the current obj.
    
    Parameters
    ----------
    current_obj : tuple
        Attributes of the current object. (class_no, x,y,w,h, frame number).
    frame : np.ndarray
        Current image frame.
    past_objs : list
        List of objects. 
        Each object is a dictionary {
                                   bbox: (x,y,w,h), 
                                   id: 0, 
                                   roi_hist: get_roi_hist(),
                                   frame_no: last frame no,
                                }
    exclude_ids : list
        Will not consider objects with listed ids as a matching object.

    Returns
    -------
    int
        Index of the best matching object.
    """
    # calculate histogram of pred
    px,py,pw,ph = int(current_obj[1][0][0]),int(current_obj[1][0][1]), int(current_obj[1][0][2]), int(current_obj[1][0][3])
    hist_pred = get_roi_hist(px,py,pw,ph, frame, padding=0.2)

    # keep track of which past object is the most similar to the current
    best_idx = None
    best_prob = 0
    best_distance = 50

    # for each past pred of same class
    for idx, ppc in enumerate(past_objs):
        # get prob if pred belongs to any past pred
        roi_hsv = cv2.cvtColor(get_roi(*ppc['bbox'], frame, 0.2), cv2.COLOR_BGR2HSV)
        prob = cv2.calcBackProject([roi_hsv], [0], hist_pred, [0,180], 1)

        sum_prob = np.sum(prob)
        
        # get dist betw bboxes
        d = get_dist_betw_bbox(ppc["bbox"], (px,py,pw,ph))

        # if past object is similar enough & is recent
        if not ppc['id'] in exclude_ids and sum_prob > max(100, best_prob*0.7) and frame_count - ppc["frame_no"] < fps:

            # if distance is lower, consider it as the best matching
            if d < best_distance:
                best_idx = idx
                best_prob = sum_prob
                best_distance = d

    return best_idx

def get_dist_betw_bbox(bbox1:tuple, bbox2:tuple):
    """
    Returns distance between bboxes.

    Parameters
    ----------
    bbox1 : tuple
        xywh format, where xy are the centre coordinates of the bbox.
    bbox2 : tuple
        xywh format, where xy are the centre coordinates of the bbox.

    Returns
    -------
    float
        Distance between the two bboxes centres.
    """
    x0,y0 = bbox1[0],bbox1[1]
    x1,y1 = bbox2[0],bbox2[1]
    return np.sqrt((x0-x1)**2 + (y0-y1)**2)

def get_roi(x:int, y:int, w:int, h:int, frame: np.ndarray, padding:float=0.0) -> np.ndarray:
    """
    Returns region of interest (ROI).

    Parameters
    ----------
    x : int
        x-coordinate of bbox centre.
    y : int
        x-coordinate of bbox centre.
    w : int
        Width of bbox.
    h : int
        Height of bbox.
    padding: float = 0.0
        To ignore the outer area of the bbox. 

    Returns
    -------
    np.ndarray
        Region of interest as an array.
    """
    roi = frame[y-round(h*padding):y+round(h*padding), x-round(w*padding):x+round(w*padding)]

    return roi

def get_roi_hist(x:int, y:int, w:int, h:int, frame: np.ndarray, padding:float=0.0) -> np.ndarray:
    """
    Returns region of interest (ROI).

    Parameters
    ----------
    x : int
        x-coordinate of bbox centre.
    y : int
        x-coordinate of bbox centre.
    w : int
        Width of bbox.
    y : int
        Height of bbox.
    padding: float = 0.0
        To ignore the outer area of the bbox. 

    Returns
    -------
    np.ndarray
        Histogram of region of interest as an array.
    """
    roi = frame[y-round(h*padding):y+round(h*padding), x-round(w*padding):x+round(w*padding)]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

    return roi_hist


if __name__ == "__main__":
    args = parseargs()

    cap = cv2.VideoCapture(args.video)
    w,h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    past_pred = {i:[] for i in range(21)}      # cache to store past predicton bboxes. 
                        # {class: 
                        #   [
                        #       {
                        #           bbox: (x,y,w,h), 
                        #           id: 0, 
                        #           roi_hist: get_roi_hist(),
                        #           frame_no: last frame no,
                        #       },
                        #   ]

    obj_count = 0       # to help assign id
    frame_count = 0     # keep track of time

    model = YOLO("best.pt")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    writer = cv2.VideoWriter("vid4_YOLOmanual.avi", fourcc, 30.0, (800,800))

    while True:
        ret, frame = cap.read()
        key = cv2.waitKey(20) & 0xFF
        frame_count += 1

        if key == ord("q"):
                break

        if ret == True:

            # preprocess frame
            frame = resize_image(frame, size=(800,800))
            # frame = preprocess_img(frame)

            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            current_pred = []   # to store objects that the model detected
            current_frame_ids = [] # to store object ids that appeared in current frame

            # pass image into model for prediction
            results = model(frame)

            for r in results:
                for b in r.boxes:
                    # assume pred = [(class_no, x,y,w,h, frame number), ...]
                    current_pred.append((int(b.cls), b.xywh, frame_count))

            # print(current_pred)
            
            # go through every object found by model
            for c_pred in current_pred:
                # calculate histogram of pred
                px,py,pw,ph = int(c_pred[1][0][0]),int(c_pred[1][0][1]), int(c_pred[1][0][2]), int(c_pred[1][0][3])
                hist_pred = get_roi_hist(px,py,pw,ph, frame, padding=0.2)

                # get past pred of same class
                past_pred_cls = past_pred[c_pred[0]]

                # if past pred of same class not empty:
                if past_pred_cls != []:
                    best_idx = get_best_matching_obj(c_pred, frame, past_pred_cls, exclude_ids=current_frame_ids)

                    # if there's a match to a past object, update the past object data
                    if best_idx is not None:
                        past_pred[c_pred[0]][best_idx]["frame_no"] = frame_count
                        past_pred[c_pred[0]][best_idx]["bbox"] = (px,py,pw,ph)
                        past_pred[c_pred[0]][best_idx]["roi_hist"] = hist_pred
                        obj_id = past_pred[c_pred[0]][best_idx]["id"]
                        draw_bbox(frame, c_pred[0], (px,py,pw,ph), obj_id)
                        current_frame_ids.append(obj_id)

                    # otherwise, create a new object 
                    else:
                        print("New object created.",obj_count)
                        # assign new id, add to past pred
                        dd = {
                            "bbox": (px,py,pw,ph), 
                            "id": obj_count, 
                            "roi_hist": hist_pred, 
                            "frame_no": frame_count
                        }
                        past_pred[c_pred[0]].append(dd)
                        draw_bbox(frame, c_pred[0], (px,py,pw,ph), obj_count)
                        current_frame_ids.append(obj_count)
                        obj_count += 1

                # if there are no existing objects for that class
                else:
                    print("New object created.",obj_count)
                    # assign new id, add to past pred
                    dd = {
                        "bbox": (px,py,pw,ph), 
                        "id": obj_count, 
                        "roi_hist": hist_pred, 
                        "frame_no": frame_count
                    }
                    past_pred[c_pred[0]].append(dd)
                    draw_bbox(frame, c_pred[0], (px,py,pw,ph), obj_count)
                    current_frame_ids.append(obj_count)
                    obj_count += 1
                
            # write and save video
            writer.write(frame)
            cv2.imshow("Tracking", frame)
            
        else:
            break

    print("total objects found:",obj_count)
    print("total frames:", frame_count)
    print("Saved video.")
    cap.release()
    writer.release()
    cv2.destroyAllWindows()