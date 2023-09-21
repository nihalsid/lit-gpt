import numpy as np
import networkx as nx

newface_token = 0
stopface_token = 1


def get_shifted_sequence(sequence):
    non_special = np.flatnonzero(np.isin(sequence, [0, 1, 2], invert=True))
    if non_special.shape[0] > 0:
        idx = non_special[0]
        val = sequence[idx]
        sequence[non_special] -= (val - 3)
    return sequence


def read_faces(text):
    all_lines = text.splitlines()
    all_face_lines = [x for x in all_lines if x.startswith('f ')]
    # print(all_face_lines)
    all_faces = [[int(y.split('/')[0]) - 1 for y in x.strip().split(' ')[1:]] for x in all_face_lines]
    return all_faces


def read_vertices(text):
    all_lines = text.splitlines()
    all_vertex_lines = [x for x in all_lines if x.startswith('v ')]
    # print(all_vertex_lines)
    all_vertices = np.array([[float(y) for y in x.strip().split(' ')[1:]] for x in all_vertex_lines])
    assert all_vertices.shape[1] == 3, 'vertices should have 3 coordinates'
    return all_vertices


def quantize_soup(vertices_, faces_, max_vertices, max_faces, num_tokens):
    def face_to_cycles(face):
        """Find cycles in face."""
        g = nx.Graph()
        for v in range(len(face) - 1):
            g.add_edge(face[v], face[v + 1])
        g.add_edge(face[-1], face[0])
        return list(nx.cycle_basis(g))

    vertices = np.clip((vertices_ + 0.5), 0, 1) * num_tokens  # type: ignore
    vertices_quantized_ = vertices.round().astype(int)

    if vertices_quantized_.shape[0] > max_vertices:
        raise ValueError("Vertices exceed max vertices:", vertices_quantized_.shape[0], max_vertices)
    if len([x for fl in faces_ for x in fl]) > max_faces:
        raise ValueError("Faces exceed max faces:", len([x for fl in faces_ for x in fl]), max_faces)

    vertices_quantized_ = vertices_quantized_[:, [2, 0, 1]]
    vertices_quantized, unique_inverse = np.unique(vertices_quantized_, axis=0, return_inverse=True)

    sort_inds = np.lexsort(vertices_quantized.T)

    vertices_quantized = vertices_quantized[sort_inds]
    vertices_quantized = np.stack([vertices_quantized[:, 2], vertices_quantized[:, 1], vertices_quantized[:, 0]], axis=-1)

    # Re-index faces and tris to re-ordered vertices.
    faces = [np.argsort(sort_inds)[unique_inverse[f]] for f in faces_]
    # Merging duplicate vertices and re-indexing the faces causes some faces to
    # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
    # sub-faces.
    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
            # Only append faces with more than two verts.
            if c_length > 2:
                d = np.argmin(c)
                # Cyclically permute faces just that first index is the smallest.
                sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
    faces = sub_faces
    # Sort faces by lowest vertex indices. If two faces have the same lowest
    # index then sort by next lowest and so on.
    faces.sort(key=lambda f: tuple(sorted(f)))

    # After removing degenerate faces some vertices are now unreferenced.
    # Remove these.
    num_verts = vertices_quantized.shape[0]
    vert_connected = np.equal(
        np.arange(num_verts)[:, None], np.hstack(faces)[None]).any(axis=-1)
    vertices_quantized = vertices_quantized[vert_connected]
    # Re-index faces and tris to re-ordered vertices.
    vert_indices = (
            np.arange(num_verts) - np.cumsum(1 - vert_connected.astype('int')))
    faces = [vert_indices[f].tolist() for f in faces]
    vertices_quantized = vertices_quantized + 3  # make space for the 3 special tokens
    soup_sequence = []
    for fi, face in enumerate(faces):
        soup_sequence.append(newface_token)
        for vi, vidx in enumerate(face):
            soup_sequence.extend(vertices_quantized[vidx, :].tolist())

    return np.array(soup_sequence)


def scale_vertices(vertices, x_lims=(0.75, 1.25), y_lims=(0.75, 1.25), z_lims=(0.75, 1.25)):
    # scale x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    vertices = np.stack([vertices[:, 0] * x, vertices[:, 1] * y, vertices[:, 2] * z], axis=-1)
    # scale back to unit cube
    return vertices


def shift_vertices(vertices, x_lims=(-0.1, 0.1), y_lims=(-0.1, 0.1), z_lims=(-0.075, 0.075)):
    # shift x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    x = max(min(x, 0.5 - vertices[:, 0].max()), -0.5 - vertices[:, 0].min())
    y = max(min(y, 0.5 - vertices[:, 1].max()), -0.5 - vertices[:, 1].min())
    z = max(min(z, 0.5 - vertices[:, 2].max()), -0.5 - vertices[:, 2].min())
    vertices = np.stack([vertices[:, 0] + x, vertices[:, 1] + y, vertices[:, 2] + z], axis=-1)
    return vertices


def normalize_vertices(vertices):
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])  # type: ignore
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    return vertices


def to_soup_sequence(output, tokenizer, fill_errors=False):
    output_text = tokenizer.decode(output).split(' ')
    soup_sequence = []
    for i in range(len(output_text)):
        if output_text[i].isdecimal():
            soup_sequence.append(int(output_text[i]))
        if fill_errors:
            soup_sequence.append(-1)
    return soup_sequence


def ngonsoup_sequence_to_mesh(output, tokenizer, num_tokens):
    soup_sequence = to_soup_sequence(output, tokenizer)
    end = len(soup_sequence)
    soup_sequence = soup_sequence[:end]
    vertices_q = []
    face_ctr = 0
    faces = []
    current_subsequence = []
    for i in range(len(soup_sequence)):
        if soup_sequence[i] == newface_token:
            current_subsequence = current_subsequence[:len(current_subsequence) // 3 * 3]
            if len(current_subsequence) == 9:
                vertices_q.append(np.array(current_subsequence).reshape(-1, 3))
                faces.append([x for x in range(face_ctr, face_ctr + len(current_subsequence) // 3)])
                face_ctr += (len(current_subsequence) // 3)
            current_subsequence = []
        else:
            current_subsequence.append(soup_sequence[i] - 3)

    current_subsequence = current_subsequence[:len(current_subsequence) // 3 * 3]
    if len(current_subsequence) == 9:
        vertices_q.append(np.array(current_subsequence).reshape(-1, 3))
        faces.append([x for x in range(face_ctr, face_ctr + len(current_subsequence) // 3)])
        face_ctr += (len(current_subsequence) // 3)

    vertices = np.vstack(vertices_q) / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)
    return vertices, faces


def plot_vertices_and_faces(vertices, faces, output_path):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ngons = [[vertices[v, :].tolist() for v in f] for f in faces]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.xlim(-0.35, 0.35)
    plt.ylim(-0.35, 0.35)
    # Don't mess with the limits!
    plt.autoscale(False)
    ax.set_axis_off()
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='black', s=10)
    polygon_collection = Poly3DCollection(ngons)
    polygon_collection.set_alpha(0.3)
    polygon_collection.set_color('b')
    ax.add_collection(polygon_collection)
    ax.set_zlim(-0.35, 0.35)
    ax.view_init(25, -120, 0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close("all")
