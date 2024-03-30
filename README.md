# Implementing-Basic-ANPR
This project aims to implement basic ANPR(Automatic Number Plate Recognition system) using OpenCV and Tesseract-OCR in Python. The program extracts the number plate information from a picture or a static videoframe and stores it in CSV file with date of entry.

# What is ANPR ?
Automatic number-plate recognition is a technology that uses optical character recognition on images to read vehicle registration plates to create vehicle location data. It can use existing closed-circuit television, road-rule enforcement cameras, or cameras specifically designed for the task.
ANPR is an advanced version of what I am going to show you in this article. Here, we will be implementing the software part. The hardware part can be implemented using CCTV cameras and Raspberry Pi.

# Steps involved :
1. Read the original image
2. Resize the image
3. Convert it to grayscale.
4. Apply Bilateral Filter.
5. Identify and store the Canny edges. 
6. Find the contours in from the edges detected and sort the top 30 contours.
7. Get the perimeter of each contour and select those with 4 corners.
8. Mask all other parts of the image and show the final image.
9. Read the text using Tesseract OCR.
10. Append to csv file.


# Dependencies
#### On Linux
```
sudo apt-get update
sudo apt-get install libleptonica-dev tesseract-ocr tesseract-ocr-dev libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn
```

#### On Mac
```
brew install tesseract
```

#### On Windows
download binary from https://github.com/UB-Mannheim/tesseract/wiki. then add pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe' to your script.

Then you should install python package using pip:
```
pip install tesseract
pip install tesseract-ocr
```

references: https://pypi.org/project/pytesseract/ (INSTALLATION section) and https://tesseract-ocr.github.io/tessdoc/Installation.html