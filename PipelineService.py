import Application as mlapp
from flask import Flask, url_for
from flask_socketio import SocketIO, emit
from jinja2 import Template
import json

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

@app.route("/score/<job_id>")
def show_score(job_id):
    return GetTemplate("score").render(job_id=job_id)



def _onitemadded_(item):
    emit(mlapp.pipeline.ItemAddedEvent, item)

def _onitemremoved_(item):
    emit(mlapp.pipeline.ItemRemovedEvent, item)

def _onschedulechanged_(schedule):
    sch_dict = {}
    sch_dict["jobs"] = []
    for sch in schedule:
        sch_dict["jobs"].append((len(sch_dict["jobs"])+1, sch[0].Name, sch[1].Name))
    json_sch = json.dumps(sch_dict)
    emit(mlapp.pipeline.ScheduleChangedEvent, json_sch)

mlapp.pipeline.Subscribe(mlapp.pipeline.ItemAddedEvent, _onitemadded_)
mlapp.pipeline.Subscribe(mlapp.pipeline.ItemRemovedEvent, _onitemremoved_)
mlapp.pipeline.Subscribe(mlapp.pipeline.ScheduleChangedEvent, _onschedulechanged_)


@socketio.on("addToSession")
def _addtosession_(item):
    mlapp.pipeline.Add(item)

@socketio.on("removeFromSession")
def _removefromsession_(item):
    mlapp.pipeline.Remove(item)

if __name__ == "__main__":
    socketio.run(app)