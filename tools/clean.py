# Deletes old model weights of failed or crashed runs to minimize storage usage

import wandb
import os

api = wandb.Api(timeout=300)
runs = api.runs("username/project_name")

for r in runs:
    model_dir = os.path.join("logs", r.id)
    if os.path.exists(model_dir) and not r.state == "running" and ("invalid" in r.tags or r.state == "crashed" or r.state == "failed"):
        print("Deleting [{}][{}][{}][{}]".format(r.id, r.state, model_dir, r.tags))
        for item in os.listdir(model_dir):
            if item.endswith(".ckpt"):
                os.remove(os.path.join(model_dir, item))