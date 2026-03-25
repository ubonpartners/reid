import ultralytics
from functools import partial
import stuff
from ultralytics.nn.modules.head import ReIDAdapter
import torch
import numpy as np

def on_predict_start(predictor: object, persist: bool = False) -> None:
            predictor.save_feats = True
            predictor.expanded_feats = True
            predictor._feats = None
            # Register hooks to extract input and output of Detect layer
            def pre_hook(module, input):
                predictor._feats = [t.clone() for t in input[0]]
            def post_hook(module, input, output):
                predictor._feats2 = output[0].clone()
            predictor.model.model.model[-1].register_forward_pre_hook(pre_hook)
            predictor.model.model.model[-1].register_forward_hook(post_hook)

model=ultralytics.YOLO("/mldata/models/v8/pt/yolo11l-v8r-130825.pt")
model.add_callback("on_predict_start", partial(on_predict_start, persist=False))

print("============= RUNNING ========")
r=model("/mldata/image/arrest2_640x640.jpg", conf=0.935)
print("============  OK ============")
cn=[]
for c in model.names:
    cn.append(model.names[c])
print(cn)
dets=stuff.yolo_results_to_dets(r[0], yolo_class_names=cn, class_names=cn)

reid_model_pt="/mldata/reid/model/reid_model-v8-130825.pth"
state_dict = torch.load(reid_model_pt, map_location='cpu')
reid_model = ReIDAdapter(in_dim=575, hidden1=160, hidden2=192, emb=80).to('cuda')
reid_model.load_state_dict(state_dict)
reid_model.eval()
reid_model.reid_layer_checksums()
exit()

for d in dets:
    if not "feats" in d:
        continue
    f=[d["feats"].cpu()]
    f=np.array(f)

    vals = f[0]

    # print 20 values per line
    for i in range(0, len(vals), 20):
        print(" ".join(f"{v:.4f}" for v in vals[i:i+20]))


    ft = torch.tensor(f.astype(np.float32)).to("cuda")
    with torch.no_grad():
         embedding = reid_model(ft).cpu().numpy()
    embedding=embedding[0]
    for i in range(80):
        print(f" {i} {embedding[i]} {d['reid_vector'][i]}")
    #print(embedding, d["reid_vector"])
