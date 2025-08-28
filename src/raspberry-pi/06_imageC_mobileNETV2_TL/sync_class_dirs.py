#!/usr/bin/env python3
# Make train/ and val/ have the SAME set of class directories
import os

DATA_DIR = os.path.join("..", "..", "..", "data", "openimages")  # <-- change if needed

train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "val")

def class_set(root):
    return {d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}

train_classes = class_set(train_dir)
val_classes   = class_set(val_dir)
union = sorted(train_classes | val_classes)

made = []
for split_dir, name in [(train_dir, "train"), (val_dir, "val")]:
    for cls in union:
        p = os.path.join(split_dir, cls)
        if not os.path.isdir(p):
            os.makedirs(p, exist_ok=True)
            made.append((name, cls))

print(f"[INFO] train classes: {len(train_classes)}  val classes: {len(val_classes)}  union: {len(union)}")
print(f"[INFO] Created {len(made)} empty class dirs to sync splits.")
