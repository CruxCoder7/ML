import matplotlib.pyplot as plt
import io
from flask import Response, Flask
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np

import pickle

model = pickle.load(open(
    r"C:\Programming\ML-Tech-Lab\Flask-Decision Tree\Flask-Decision Tree\model.pkl", 'rb'))
app = Flask(__name__)


@app.route('/plot')
def plot():
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(35, 30))
    _ = tree.plot_tree(model, filled=True)

    # Save the plot to a file in memory
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)

    # Serve the file to the client
    return Response(img.getvalue(), mimetype='image/png')


if __name__ == "__main__":
    app.run(debug=True)

# add /plot to the site after running the local host
