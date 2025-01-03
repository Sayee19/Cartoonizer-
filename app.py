import os
import io
import uuid
import sys
import yaml
import traceback

with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)

sys.path.insert(0, './white_box_cartoonizer/')

import cv2
from flask import Flask, render_template, make_response, flash
import flask
from PIL import Image
import numpy as np
import skvideo.io
if opts['colab-mode']:
    from flask_ngrok import run_with_ngrok #to run the application on colab using ngrok


from cartoonize import WB_Cartoonize

###new stuff###
def process_video(input_video_path, output_video_path, frame_rate):
    import cv2
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cartoon_frame = wb_cartoonizer.infer(frame)  # White-Box Cartoonizer processing
        out.write(cartoon_frame)

    cap.release()
    out.release()

    return output_video_path
###new stuff end###

if not opts['run_local']:
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        from gcloud_utils import upload_blob, generate_signed_url, delete_blob, download_video
    else:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS not set in environment variables")
    from video_api import api_request
    # Algorithmia (GPU inference)
    import Algorithmia

app = Flask(__name__)
if opts['colab-mode']:
    run_with_ngrok(app)   #starts ngrok when the app is run

app.config['UPLOAD_FOLDER_VIDEOS'] = 'static/uploaded_videos'
app.config['CARTOONIZED_FOLDER'] = 'static/cartoonized_images'

app.config['OPTS'] = opts

## Init Cartoonizer and load its weights 
wb_cartoonizer = WB_Cartoonize(os.path.abspath("white_box_cartoonizer/saved_models/"), opts['gpu'])

def convert_bytes_to_image(img_bytes):
    """Convert bytes to numpy array

    Args:
        img_bytes (bytes): Image bytes read from flask.

    Returns:
        [numpy array]: Image numpy array
    """
    
    pil_image = Image.open(io.BytesIO(img_bytes))
    if pil_image.mode=="RGBA":
        image = Image.new("RGB", pil_image.size, (255,255,255))
        image.paste(pil_image, mask=pil_image.split()[3])
    else:
        image = pil_image.convert('RGB')
    
    image = np.array(image)
    
    return image

@app.route('/')
@app.route('/cartoonize', methods=["POST", "GET"])
def cartoonize():
    opts = app.config['OPTS']
    if flask.request.method == 'POST':
        try:
            # Handle image uploads
            if flask.request.files.get('image'):
                img = flask.request.files["image"].read()
                image = convert_bytes_to_image(img)
                img_name = str(uuid.uuid4())
                cartoon_image = wb_cartoonizer.infer(image)
                cartoonized_img_name = os.path.join(app.config['CARTOONIZED_FOLDER'], img_name + ".jpg")
                cv2.imwrite(cartoonized_img_name, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))
                return render_template("index_cartoonized.html", cartoonized_image=cartoonized_img_name)

            # Handle video uploads
            if flask.request.files.get('video'):
                video = flask.request.files["video"]
                filename = str(uuid.uuid4()) + ".mp4"
                original_video_path = os.path.join(app.config['UPLOAD_FOLDER_VIDEOS'], filename)
                video.save(original_video_path)

                cartoonized_video_path = os.path.join(app.config['UPLOAD_FOLDER_VIDEOS'], filename.split(".")[0] + "_cartoonized.mp4")
                frame_rate = 30  # Adjust based on your requirement

                # Call the local video processing function
                process_video(original_video_path, cartoonized_video_path, frame_rate)

                return render_template("index_cartoonized.html", cartoonized_video=cartoonized_video_path)

        except Exception as e:
            import traceback
            traceback.print_exc()
            flask.flash(f"An error occurred: {str(e)}")
            return render_template("index_cartoonized.html")

    return render_template("index_cartoonized.html")


if __name__ == "__main__":
    # Commemnt the below line to run the Appication on Google Colab using ngrok
    if opts['colab-mode']:
        app.run()
    else:
        app.run(debug=False, host='127.0.0.1', port=int(os.environ.get('PORT', 8080)))
