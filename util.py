import torch, math
import numpy as np
import torchvision
from segment import segment
import os

def projection(x=0.1, n=1.0, f=50.0):
    return torch.tensor([[n/x,    0.,            0.,              0],
                     [  0., n/-x,            0.,              0],
                     [  0.,    0., -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0.,    0.,           -1.,              0.]]).float()

def translate(x=0., y=0., z=0.):
    return torch.tensor([[1., 0., 0., x],
                     [0., 1., 0., y],
                     [0., 0., 1, z],
                     [0., 0., 0., 1.]]).float()


def rotate_z(a):
    s, c = math.sin(a), math.cos(a)
    return torch.tensor([[c,  s, 0, 0],
                        [-s,  c, 0, 0],
                        [0, 0, 1., 0],
                        [0,  0, 0, 1]]).float()

def rotate_x(a):
    s, c = math.sin(a), math.cos(a)
    return torch.tensor([[1,  0, 0, 0],
                     [0,  c, s, 0],
                     [0, -s, c, 0],
                     [0,  0, 0, 1]]).float()

def rotate_y(a):
    s, c = math.sin(a), math.cos(a)
    return torch.tensor([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]]).float()

_glfw_window = None
def display_image(image, zoom=None, size=None, title=None): # HWC
    # Import OpenGL and glfw.
    import OpenGL.GL as gl
    import glfw

    # Zoom image if requested.
    image = np.asarray(image)
    if size is not None:
        assert zoom is None
        zoom = max(1, size // image.shape[0])
    if zoom is not None:
        image = image.repeat(zoom, axis=0).repeat(zoom, axis=1)
    height, width, channels = image.shape

    # Initialize window.
    if title is None:
        title = 'Debug window'
    global _glfw_window
    if _glfw_window is None:
        glfw.init()
        _glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(_glfw_window)
        glfw.show_window(_glfw_window)
        glfw.swap_interval(0)
    else:
        glfw.make_context_current(_glfw_window)
        glfw.set_window_title(_glfw_window, title)
        glfw.set_window_size(_glfw_window, width, height)

    # Update window.
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {'uint8': gl.GL_UNSIGNED_BYTE, 'float32': gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(_glfw_window)
    if glfw.window_should_close(_glfw_window):
        return False
    return True

class ReferenceImages(torch.utils.data.Dataset):
    def __init__(self, dir, w, h):
        self.dir = dir

        self.segment_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        # self.transforms = torchvision.transforms.Resize((h,w))
        self.transforms = torchvision.transforms.Compose([
                            torchvision.transforms.ToPILImage(),
                            torchvision.transforms.Resize((h,w)),
                            torchvision.transforms.ToTensor()
                        ])

        self.images = []

        # presegment the images
        for image in os.listdir(dir):
            im = image.split('.')
            rot = im[1]

            self.images.append((self.transform(os.path.join(self.dir, image)), math.radians(float(rot))))



    def transform(self, im: str):
        return self.transforms(segment([im], model=self.segment_model, use_cuda=True, background_color=0)[0].cpu())

    def __getitem__(self, idx):
        """
        im = self.images[idx].split('.')
        rot = im[1]

        return self.transform(os.path.join(self.dir, self.images[idx])), math.radians(float(rot))
        """
        return self.images[idx]

    def __len__(self):
        return len(self.images)


def load_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        for s in f:
            l = s.strip('\n')
            if len(l) == 0:
                continue
            if l[0] == 'v':
                parts = l.split(' ')
                vertices.append([float(x) for x in parts[1:]])
            if l[0] == 'f':
                parts = l.split(' ')
                faces.append([int(x) for x in parts[1:]])
    # make sure faces is  0 indexed
    return {'pos_idx': np.array(faces) - 1, 'vtx_pos': np.array(vertices)}

def compute_laplace_matrix(verts, faces):
    print(faces.min(), faces.max())
    print(verts.shape)
    L = torch.zeros(verts.shape[0], verts.shape[0])
    """
    for vi in range(verts.shape[0]):
        # find neighbors
        for fi in range(faces.shape[0]):
            if faces[fi][0] == vi or faces[fi][1] == vi or faces[fi][2] == vi:
                L[vi][faces[fi][0]] = 1
                L[vi][faces[fi][1]] = 1
                L[vi][faces[fi][2]] = 1
        L[vi][vi] = 0
        L[vi] /= L[vi].sum()
    """
    for fi in range(faces.shape[0]):
        for vi in faces[fi]:
            L[vi][faces[fi][0]] = 1
            L[vi][faces[fi][1]] = 1
            L[vi][faces[fi][2]] = 1
    
    for vi in range(verts.shape[0]):
        L[vi][vi] = 0
        L[vi] /= L[vi].sum()
    

    #torch.save(L, 'laplace.pt')
    print("Finished computing laplacian")

    return L

    
def compute_curvature(verts, laplace):
    return verts - torch.matmul(laplace, verts)