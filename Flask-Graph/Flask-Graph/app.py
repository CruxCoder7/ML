import matplotlib.pyplot as plt
import io
from flask import Response, Flask

import numpy as np

history = np.load(
    r"C:\Programming\ML-Tech-Lab\Flask-Graph\Flask-Graph\my_history.npy", allow_pickle='TRUE').item()


app = Flask(__name__)


@app.route('/plot')
def plot():
    fig, ax = plt.subplots()
    ax.plot(history['accuracy'], label='accuracy')

    # Save the plot to a file in memory
    img = io.BytesIO()
    fig.savefig(img)
    # Serve the file to the client
    return Response(img.getvalue(), mimetype='image/png')


if __name__ == "__main__":
    app.run(port=6546, debug=True)
