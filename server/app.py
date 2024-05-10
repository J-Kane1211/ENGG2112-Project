from flask import Flask
from flask import render_template
from flask import request



if __name__ == '__main__' and __package__ is None:
    import os
    from os import sys, path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import validate
    print(validate)
    app = Flask(__name__)
    app.run()
    

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)


measurements = []

@app.route('/input', methods=['GET','POST'])
def get_user_input():
    global measurements
    if request.method == 'POST':
        #handle the data here, and do redirect
        for i in range(10):
            measurements[i][1] = request.form['number_{}'.format(i)]
        print(measurements)
        
        return "<p>POST logged</p>"
    else:
        #create user input table, with some default data
        title = "Please input your data"
        if not measurements:
            for i in range(10):
                measurements.append(['number_{}'.format(i), i])
        return render_template("input.html", title=title, measurements=measurements)
