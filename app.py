from flask import Flask, request

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return 'Hello World'


@app.route('/xyz', methods=['GET'])
def xyz():
    return 'Hello xyz'


@app.route('/user', methods=['GET'])
def user():
    name = request.args['name']
    return f'Hello {name}!!!'


@app.route('/user2', methods=['POST'])
def user2():
    name = request.form['name']
    return f'Hello {name}!!!'


if __name__ == '__main__':
    app.run(port=8000, debug=True)
