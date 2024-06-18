import gradio as gr
from face_emotion_pipeline import process_video, process_image

def process_file(file, is_video , skip = 1  , add_audio = True):
    # print("==========>", is_video)
    print("==========>", skip)
    print("==========>", add_audio)
    input_path = file.name
    output_path = "output." + ("mp4" if is_video else "png")
    if is_video == True:
        process_video(input_path, output_path , skip = round(skip)  , add_audio = add_audio)
    else:
        process_image(input_path, output_path)
    return output_path

iface = gr.Interface(
    fn=process_file,
    inputs=[gr.File(label="Upload File"), gr.Checkbox(label="Is Video?") , gr.Slider(1, 20, 1.0, value=5, label="Frame Skip"), gr.Checkbox(label="Add Audio?")],
    outputs=gr.File(label="Processed File"),
    title="Face Emotion Detection",
    description="""Upload an image or video to detect and annotate emotions. <br>
    - For videos, the detected faces will be annotated with emotions and the audio can be optionally added back.
    You can also skip some frames to speed up the process. <br>
    - For images, the detected faces will be annotated with emotions. <br>
    """
)

if __name__ == "__main__":
    iface.launch(share=True)
