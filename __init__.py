from flask import Flask, render_template, request, redirect, jsonify
from client import stable_diffusion
from sendToAirtable import add_new_record
from uploadToS3 import upload_file_using_client
app = Flask(__name__)

@app.route("/")
def accept_prompt():
    return render_template('prompt_form.html')

@app.route("/run_txt_2_img", methods=['POST'])
def run_stable_dif():
    if request.method == "POST":
        prompt = request.form['prompt']
        num_samples = request.form['num_samples']
        print('num_samples:')
        print(num_samples)
        output = stable_diffusion(prompt = prompt, num_samples = num_samples)
    return redirect(f"https://dreamstudiooutputs.s3.us-west-1.amazonaws.com/{output[0]}.png", code=302)

@app.route("/api/v1/generation", methods=['POST'])
def stable_dif_api():
    if request.method == "POST":
        params = request.form.to_dict()
        print('prompt:')
        print(params['prompt'])
        output = stable_diffusion(prompt = params['prompt'], num_samples = int(params['num_samples']))
        data = {
            "image_urls" : output
        }
    return jsonify(data)