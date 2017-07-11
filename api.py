from flask import Flask, url_for
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

if __name__ == '__main__':
    app.run()