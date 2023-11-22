import numpy as np


def load_obj(fn):
    fin = open(fn, "r")
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith("f "):
            faces.append(np.int32([item.split("/")[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)
    return v, f


def export_obj(out, v, f):
    with open(out, "w") as fout:
        for i in range(v.shape[0]):
            fout.write("v %f %f %f\n" % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write("f %d %d %d\n" % (f[i, 0], f[i, 1], f[i, 2]))
