from new.main_HR import calcHR
from new.quality import getQuality
from flask import Flask, render_template, request
from main import train, inference
import numpy as np

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/train", methods = ['GET'])
def get_trained():
	if request.method == 'GET':
		accuracy = str(train())

	return render_template("index.html", accuracy = accuracy)

@app.route("/quality", methods = ['POST'])
def get_quality():
	if request.method == 'POST':
		req = request.get_json()
	quality = getQuality(req['dir'])

	return 200 if quality else 'The image has bad quality',400

@app.route("/hr", methods = ['POST'])
def get_hr():
	if request.method == 'POST':
		req = request.get_json()
	hr = calcHR(True, req['dir'])

	return hr



@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "test/" + img.filename	
		img.save(img_path)

		p = np.array2string(inference(img_path))

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)