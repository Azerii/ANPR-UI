import numpy as np
import cv2
import pytesseract

def cars_crossing_boundary(frame: np.ndarray):
  """
  This function checks for car contours that have crossed a boundary line 
  based on the assumption that the camera is 3 meters above ground level.

  Args:
      frame: A NumPy array representing the video frame.

  Returns:
      A list of contours that potentially represent cars crossing the boundary.
  """

  # Preprocessing (grayscale, noise reduction, Canny edge detection)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  filtered = cv2.fastNlMeansDenoising(gray)
  edges = cv2.Canny(filtered, 170, 200)

  # Contour detection
  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  # Assuming an average car height of 1.5 meters and camera 3 meters high
  # Minimum height from the bottom of the frame for a car to be considered crossing
  car_bottom_threshold = int(frame.shape[0] * (1.5 + 3) / 4.5)

  # Filter contours based on bounding rectangle and bottom position
  cars_crossing = []
  for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if 1.5 < float(w) / h < 2.5 and y > car_bottom_threshold:
      cars_crossing.append(c)

  return cars_crossing

# Tesseract configuration
configr = ('-l eng --oem 1 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-%')

def process_frame(frame: np.ndarray) -> str | None:
  # Convert from coloured to Grayscale.
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2.imshow("Preprocess 1 - Grayscale Conversion", gray)
  cv2.waitKey(0)

  # Noise reduction and applying Bilateral Filter on the grayscale image.
  denoised = cv2.fastNlMeansDenoising(gray)
  filtered = cv2.bilateralFilter(denoised, 9, 15, 15)
  cv2.imshow("Preprocess 2 - Bilateral Filter", filtered)
  cv2.waitKey(0)

  # Finding edges of the grayscale image.
  edges = cv2.Canny(filtered, 170, 200)
  cv2.imshow("Preprocess 3 - Canny Edges", edges)
  cv2.waitKey(0)

  # Contour detection
  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  # cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

  boundary_y = 100
  cv2.line(frame, (0, boundary_y), (frame.shape[1], boundary_y), (0, 255, 0), 2)  # Green line

  # Number plate localization and OCR
  for c in contours:
      x, y, w, h = cv2.boundingRect(c)
      aspect_ratio = float(w) / h
      if 1.5 < aspect_ratio < 2.5:
          roi = frame[y:y + h, x:x + w]
          try:
              # Preprocess ROI for better OCR accuracy
              roi = cv2.equalizeHist(roi)
              roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

              text = pytesseract.image_to_string(roi, config=configr)
              print("Number plate:", text)
              # assert len(text) >= 7

              return text

          except:
              pass  # Handle OCR errors gracefully
