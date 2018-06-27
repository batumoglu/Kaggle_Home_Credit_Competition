import Application as mlapp
from flask import Flask, url_for
from jinja2 import Template

app = Flask(__name__, static_url_path="/static")

def GetTemplate(name):
    with open("".join(["templates/", name ,".j2"])) as template:
        return Template(template.read())

@app.route("/")
def index():
    models = mlapp.pipeline.Models
    datasets = mlapp.pipeline.Datasets
    return GetTemplate("index").render(models=models, datasets=datasets)
