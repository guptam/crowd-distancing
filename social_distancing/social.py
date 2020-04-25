import cv2
import numpy as np
import json
from homography import Homography

# file = 'mall.mp4'
file = 'town.avi'

def read_boxes_for_frame(frame_idx):
    boxes = []
    with open(f'../boxes_{file}/frame_{frame_idx}.txt', 'r') as f:
        num_lines = int(f.readline())
        for _ in range(num_lines):
            boxes.append(list (map(int, f.readline().split())) )
    return boxes

size = 500

if __name__ == "__main__":
    cap = cv2.VideoCapture(f'./{file}')

    with open(f'calib_{file}.json') as f:
        calib = json.load(f)
    
    image_points = np.array(calib["image"])
    map_points = np.array(calib["map"])
    dist_coords = calib["dist"]

    scale = calib["scale"]

    homography = Homography(image_points, size * (scale / 2) + size * scale + map_points * size * scale)

    dist_coords = [ homography.project(dist_coords[0]), homography.project(dist_coords[1]) ]
    min_dist = np.linalg.norm(dist_coords[0] - dist_coords[1])

    frame_idx = 0

    while True:
        r, img = cap.read()
        img = cv2.resize(img, (1280, 720))

        if frame_idx == 0:
            cv2.imwrite(file + '.jpg', img)

        try:
            boxes = read_boxes_for_frame(frame_idx)
        except:
            cap.release()
            cap = cv2.VideoCapture(f'./{file}')
            frame_idx = 0
            continue
        
        points = []
        points_norm = []

        for i in range(len(boxes)):
            box = boxes[i]
            
            mid = ( int((box[1] + box[3]) / 2), int(box[2]) )

            p = homography.project(np.array(mid))
        
            points.append(p)

            cv2.circle(img, mid, 4, (0,0,255), -1)
            cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(230, 180, 48),2)
        
        projection = np.zeros((500, 500, 3), dtype=np.uint8)

        close_points = []

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                if dist < min_dist:
                    close_points.append(i)
                    close_points.append(j)


        for idx in range(len(points)):
            point = points[idx]
            if idx in close_points:
                cv2.circle(projection, (int(point[0]), int(point[1])), 3, (48, 180, 230), -1)
                box = boxes[idx]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(48, 180, 230),2)

            else:
                cv2.circle(projection, (int(point[0]), int(point[1])), 3, (230, 180, 48), -1)

        frame_idx += 1

        cv2.imshow("preview", img)
        cv2.imshow("projection", projection)
        key = cv2.waitKey(33)
        if key & 0xFF == ord('q'):
            break
