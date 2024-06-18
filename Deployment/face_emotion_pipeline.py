import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip
from retinaface import RetinaFace
from hsemotion.facial_emotions import HSEmotionRecognizer


# Initialize recognizer
recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device='cpu')
 
# Face Detection Function
def detect_faces(frame):
    """ Detect faces in the frame using RetinaFace """
    faces = RetinaFace.detect_faces(frame)
    if isinstance(faces, dict):
        face_list = []
        for key in faces.keys():
            face = faces[key]
            facial_area = face['facial_area']
            face_dict = {
                'box': (facial_area[0], facial_area[1], facial_area[2] - facial_area[0], facial_area[3] - facial_area[1])
            }
            face_list.append(face_dict)
        return face_list
    return []
 
# Annotation Function
def annotate_frame(frame, faces):
    """ Annotate the frame with recognized emotions using global recognizer """
    for face in faces:
        x, y, w, h = face['box']
        face_image = frame[y:y+h, x:x+w]  # Extract face region from frame
        emotion = classify_emotions(face_image)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
 
# Emotion Classification Function
def classify_emotions(face_image):
    """ Classify emotions for the given face image using global recognizer """
    results = recognizer.predict_emotions(face_image)
    if results:
        emotion = results[0]  # Get the most likely emotion
        print("=====>",emotion)
    else:
        emotion = 'Unknown'
    return emotion
 
# Process Video Frames
def process_video_frames(video_path, temp_output_path, frame_skip=5):
    # Load the video
    video_clip = VideoFileClip(video_path)
    fps = video_clip.fps
 
    # Initialize output video writer
    out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(video_clip.size[0]), int(video_clip.size[1])))
 
    # Iterate through frames, detect faces, and annotate emotions
    frame_count = 0
    for frame in video_clip.iter_frames():
        if frame_count % frame_skip == 0:  # Process every nth frame
            frame = np.copy(frame)  # Create a writable copy of the frame
            faces = detect_faces(frame)
            annotate_frame(frame, faces)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert RGB to BGR for OpenCV RGB2BGR
        out.write(frame)
        frame_count += 1
 
    # Release resources and cleanup
    out.release()
    cv2.destroyAllWindows()
    video_clip.close()
 
# Add Audio to Processed Video
def add_audio_to_video(original_video_path, processed_video_path, output_path):
    try:
        original_clip = VideoFileClip(original_video_path)
        processed_clip = VideoFileClip(processed_video_path)
        final_clip = processed_clip.set_audio(original_clip.audio)
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    except Exception as e:
        print(f"Error while combining with audio: {e}")
    finally:
        original_clip.close()
        processed_clip.close()
 
# Process Video
def process_video(video_path, output_path , skip = 1  , add_audio = True):
    temp_output_path = 'temp_output_video.mp4'
 
    # Process video frames and save to a temporary file
    process_video_frames(video_path, temp_output_path, frame_skip=skip)  # Adjust frame_skip as needed
 
    # Add audio to the processed video
    if add_audio:
        add_audio_to_video(video_path, temp_output_path, output_path)
    else:
        os.rename(temp_output_path, output_path)  # Rename the temporary file if audio is not needed
 
# Process Image
def process_image(input_path, output_path):
    # Ensure output path has a valid extension
    if not output_path.lower().endswith(('.jpg', '.jpeg', '.png','.heic')):
        output_path += '.jpg'  # Default to .jpg if no valid extension is found
 
    # Step 1: Read input image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Unable to read image at '{input_path}'")
        return
 
    # Step 2: Detect faces and annotate emotions
    faces = detect_faces(image)
    annotate_frame(image, faces)
 
    # Step 3: Write annotated image to output path
    cv2.imwrite(output_path, image)
 
    # Step 4: Combine input and output images horizontally
    input_image = cv2.imread(input_path)
    combined_image = cv2.hconcat([input_image, image])
    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
 
    # Step 5: Save the combined image
    combined_output_path = os.path.splitext(output_path)[0] + '_combined.jpg'
    
    cv2.imwrite(combined_output_path, combined_image)


    ###########################
    
# recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device='cpu')

# def detect_faces(frame):
#     faces = RetinaFace.detect_faces(frame)
#     if isinstance(faces, dict):
#         face_list = []
#         for key in faces.keys():
#             face = faces[key]
#             facial_area = face['facial_area']
#             face_dict = {
#                 'box': (facial_area[0], facial_area[1], facial_area[2] - facial_area[0], facial_area[3] - facial_area[1])
#             }
#             face_list.append(face_dict)
#         return face_list
#     return []

# def annotate_frame(frame, faces):
#     frame_writable = np.copy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Make a writable copy of the frame
#     for face in faces:
#         x, y, w, h = face['box']
#         face_image = frame_writable[y:y+h, x:x+w]
#         emotion = classify_emotions(face_image)
#         cv2.rectangle(frame_writable, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         cv2.putText(frame_writable, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#     return frame_writable

# def classify_emotions(face_image):
#     results = recognizer.predict_emotions(face_image)
#     if results:
#         emotion = results[0]
#     else:
#         emotion = 'Unknown'
#     return emotion

# def process_video_frames(video_path, temp_output_path, frame_skip=5):
#     video_clip = VideoFileClip(video_path)
#     fps = video_clip.fps
#     out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(video_clip.size[0]), int(video_clip.size[1])))
#     frame_count = 0
#     for frame in video_clip.iter_frames():
#         if frame_count % frame_skip == 0:
#             faces = detect_faces(frame)
#             annotated_frame = annotate_frame(frame, faces)
#             frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#         out.write(frame)
#         frame_count += 1
#     out.release()
#     cv2.destroyAllWindows()
#     video_clip.close()

# def add_audio_to_video(original_video_path, processed_video_path, output_path):
#     try:
#         original_clip = VideoFileClip(original_video_path)
#         processed_clip = VideoFileClip(processed_video_path)
#         final_clip = processed_clip.set_audio(original_clip.audio)
#         final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
#     except Exception as e:
#         print(f"Error while combining with audio: {e}")
#     finally:
#         original_clip.close()
#         processed_clip.close()

# def process_video(video_path, output_path):
#     temp_output_path = 'temp_output_video.mp4'
#     process_video_frames(video_path, temp_output_path, frame_skip=5)
#     add_audio_to_video(video_path, temp_output_path, output_path)

# def process_image(input_path, output_path):
#     image = cv2.imread(input_path)
#     if image is None:
#         print(f"Error: Unable to read image at '{input_path}'")
#         return
#     faces = detect_faces(image)
#     annotated_image = annotate_frame(image, faces)
#     cv2.imwrite(output_path, annotated_image)
#     input_image = cv2.imread(input_path)
#     combined_image = cv2.hconcat([input_image, annotated_image])
#     cv2.imwrite(output_path, combined_image)


###################################################