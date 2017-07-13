import os
from flask import Flask, render_template_string, url_for, json
app = Flask(__name__)

@app.route('/')
def api_root():
    return 'Welcome'


@app.route('/workers')
def api_articles():
    return 'List of ' + url_for('api_articles')

@app.route('/workers/<workerid>')
def api_article(workerid):
    return 'You are reading ' + workerid

@app.route('/raw_json')
def showjson():
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(SITE_ROOT, "data", "workers_output.json")
    data = json.load(open(json_url))
    
    return render_template_string('{{data}}', data=data)

if __name__ == '__main__':
    app.run()