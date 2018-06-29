import Application as mlapp
from flask import Flask, url_for
from flask_socketio import SocketIO, emit
from jinja2 import Template

app = Flask(__name__, static_url_path="/static")
socketio = SocketIO(app)


def GetTemplate(name):
    with open("".join(["templates/", name ,".j2"])) as template:
        return Template(template.read())

@app.route("/")
def index():
    models = []
    datasets = []
    for item in mlapp.pipeline.Items:
        if item.Type == "model":
            models.append(item)
        elif item.Type == "dataset":
            datasets.append(item)
    return GetTemplate("index").render(models=models, datasets=datasets)


def _onitemadded_(item):
    emit(mlapp.pipeline.ItemAddedEvent, item)

def _onitemremoved_(item):
    pass

def _onschedulechanged_(schedule):
    pass

mlapp.pipeline.Subscribe(mlapp.pipeline.ItemAddedEvent, _onitemadded_)
mlapp.pipeline.Subscribe(mlapp.pipeline.ItemRemovedEvent, _onitemremoved_)
mlapp.pipeline.Subscribe(mlapp.pipeline.ScheduleChangedEvent, _onschedulechanged_)


@socketio.on("addToSession")
def _addtosession_(item):
    mlapp.pipeline.Add(item)


if __name__ == "__main__":
    socketio.run(app)