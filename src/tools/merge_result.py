import os
import json


src_dir = "/extra/code/mvpose/result-loaded/geometry-mean/Shelf"

merged_results = {}
for name in sorted(os.listdir(src_dir)):
    with open(os.path.join(src_dir, name), "r") as f:
        results = json.load(f)
    if not merged_results:
        merged_results.update(results)
        continue

    merged_results["res_3d"].update(results["res_3d"])
    for camera_id in merged_results["res_2d"].keys():
        merged_results["res_2d"][camera_id].update(results["res_2d"][camera_id])


with open(os.path.join(src_dir, "merged.json"), "w") as f:
    json.dump(merged_results, f)

