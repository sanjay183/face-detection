
# from flask import Flask, render_template, request
# import cv2
# import os

# app = Flask(__name__)

# def detect_faces(image):
#     # Load pre-trained face detection model
#     face_cascade = cv2.CascadeClassifier('project/haarcascade_frontalface_default.xml')

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Perform face detection
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Draw rectangles around detected faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     return image

# @app.route('/')
# def index():
#     return render_template('data.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     # Retrieve the uploaded file
#     uploaded_file = request.files['file']

#     # Save the uploaded file to a temporary location
#     image_path = 'temp.jpg'
#     uploaded_file.save(image_path)

#     # Load the image using OpenCV
#     image = cv2.imread(image_path)

#     # Perform face detection
#     output_image = detect_faces(image)
    
    
#     # Display the output image
#     output_path='static/DSC_0252.JPG'
#     cv2.imwrite(output_path, output_image)

#     return render_template('result.html')

# if __name__ == '__main__':

#     app.run(debug=True)


from flask import Flask, render_template, request
import cv2
import os
from github import Github

app = Flask(__name__)

def detect_faces(image):
    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier('project/haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image

@app.route('/')
def index():
    return render_template('data.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Retrieve the uploaded file
    uploaded_file = request.files['file']

    # Save the uploaded file to a temporary location
    image_path = 'temp.jpg'
    uploaded_file.save(image_path)

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Perform face detection
    output_image = detect_faces(image)

    # Save the output image
    output_path = 'project/static/DSC_0252.JPG'
    cv2.imwrite(output_path, output_image)

    # Save the output image to GitHub
    access_token = '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'  # Replace with your GitHub access token
    repository_name = 'face-detection'  # Replace with your GitHub repository name
    branch_name='main'
    username='sanjay183'
    g = Github(access_token)
    repo = g.get_user().get_repo(repository_name)
    with open(output_path, 'rb') as file:
        content = file.read()
    repo.create_file(output_path, "Committing output image", content, branch=branch_name)

    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
