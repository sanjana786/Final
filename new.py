import face_recognition
import os
from PIL import Image
from IPython.display import display  # For Jupyter Notebook

# Load known faces and their images from a directory
known_faces = []
known_images = []

known_faces_dir = "photo"
allowed_extensions = (".jpg", ".jpeg", ".png")

for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(allowed_extensions):
        face_image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
        face_encodings = face_recognition.face_encodings(face_image)
        
        # Check if a face was found in the image
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]  # Assuming one face per image
            known_faces.append(face_encoding)
            known_images.append(Image.open(os.path.join(known_faces_dir, filename)))

# Load an unknown face for matching
unknown_image = face_recognition.load_image_file("12.jpg")
unknown_face_encodings = face_recognition.face_encodings(unknown_image)

# Check if a face was found in the unknown image
if len(unknown_face_encodings) > 0:
    unknown_face_encoding = unknown_face_encodings[0]
    
    # Compare the unknown face to the known faces
    results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

    if True in results:
        # Match found
        match_index = results.index(True)
        matched_image = known_images[match_index]
        matched_image.show()  # Show the matched image
        print("Successful match found")
    else:
        # No match found
        print("No match found")
else:
    # No face found in the unknown image
    print("No face found in the unknown image")
