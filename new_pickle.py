import face_recognition
import os
import pickle

def get_face_encodings(directory):
    known_faces_dict = {}

    allowed_extensions = {".jpg", ".jpeg", ".png"}

    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in allowed_extensions):
            image = face_recognition.load_image_file(os.path.join(directory, filename))
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]  # Considering only the first face
                known_faces_dict[filename.split('.')[0]] = face_encoding

    return known_faces_dict

# Create a dictionary to associate face encodings with their respective images
known_faces_dict = get_face_encodings("photo")  # Replace "photo" with your directory path

# Store the known_faces_dict in a pickle file
with open('known_faces_dict.pkl', 'wb') as file:
    pickle.dump(known_faces_dict, file)
