import Application as mlapp
from flask import Flask, url_for
from jinja2 import Template

app = Flask(__name__, static_url_path="/static")

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

if __name__ == "__main__":
    app.run()