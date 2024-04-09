import cv2
import time
import pandas as pd
from process_frame import process_frame, cars_crossing_boundary

if __name__ == '__main__':
    cap = cv2.VideoCapture('C:\\Users\\User A\\csc\\anpr\\assets\\video.mp4')  # Use 0 for webcam
    processing_car = False

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Warning: No stream available")
            break

        cv2.imshow("Video", frame)
        if not processing_car and any(cars_crossing_boundary(frame)):
            processing_car = True
            plate_number = process_frame(frame)

            #The extracted data is stored in a data file.
            data = {'Date': [time.asctime(time.localtime(time.time()))],
                'Vehicle_number': [plate_number]}

            df = pd.DataFrame(data, columns = ['Date', 'Vehicle_number'])
            df.to_csv('C:\\Users\\User A\\csc\\anpr\\assets\\Dataset_VehicleNo.csv')

            cv2.imshow("Video", frame)

            processing_car = False

        if cv2.waitKey(1) == ord('q'):
            processing_car = False
            break

    cap.release()
    cv2.destroyAllWindows()