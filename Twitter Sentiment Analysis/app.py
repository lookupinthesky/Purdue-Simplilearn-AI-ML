from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
from preprocessing import TensorflowTweet
import os 
  



app = Flask(__name__)

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def render_html():
    return render_template("twitter.html")

@app.route('/', methods=['POST'])
def receive_tweet():
    #for local
    #BASE = "http://0.0.0.0:5000/" 

    BASE = 'https://twitterdocksentiment.herokuapp.com/' #for heroku
    tweet = request.form['tweet']
    params = dict()
    params['text'] = tweet
    response = requests.get(BASE + '/predict/', params=params, verify=False)

    print(f"inside receive tweet response is {response.text}")

    return render_template('twitter.html', result=response.text)

@app.route('/predict/', methods = ['GET'])
def predict():
    tweet = request.args['text']
    tfw = TensorflowTweet('models/model41.h5')
    response = tfw.predict(tweet)
    print(f"inside predict, response is {response}")
    return jsonify(response)


def predict1(tweet):
    tweet = request.args['text']
    tfw = TensorflowTweet('models/model41.h5')
    response = tfw.predict(tweet)
    print(f"inside predict, response is {response}")
    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) #for heroku
    app.run(debug=True, port=port) # for heroku

  #for local
  #  app.run(debug=True)




