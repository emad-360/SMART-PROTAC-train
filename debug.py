import torch

checkpoint = torch.load(
    "model/SE3-PROTACs_best_96.pt",
    map_location="cpu",
    weights_only=False
)

print(type(checkpoint))
print("\nKeys n Values:\n")

for key,value in checkpoint.items():
    if key=='epoch':
        print(key, value)
    else:
        print(key)
        
print("\n")

checkpoint2 = torch.load(
    "model/SE3-PROTACs_final_100.pt",
    map_location="cpu",
    weights_only=False
)

print(type(checkpoint2))
print("\nKeys n Values:\n")

for key,value in checkpoint2.items():
    if key=='epoch':
        print(key, value)
    else:
        print(key)

print("\n")
checkpoint3 = torch.load(
    "model/SE3-PROTACs_best.pt",
    map_location="cpu",
    weights_only=False
)

print("Epoch:", checkpoint3['epoch'])
print("Validation AUROC:", checkpoint3['val_auroc'])