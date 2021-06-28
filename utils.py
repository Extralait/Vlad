import glfw
import pyrr
from OpenGL.GL import glBindTexture, glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, \
    GL_TEXTURE_WRAP_T, GL_REPEAT, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,\
    glTexImage2D, GL_RGBA, GL_UNSIGNED_BYTE
from PIL import Image
import numpy as np

# for use with pygame
def load_texture(path, texture):
    glBindTexture(GL_TEXTURE_2D, texture)
    # Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # load image
    image = Image.open(path)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = image.convert("RGBA").tobytes()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    return texture


class ObjLoader:
    buffer = []

    @staticmethod
    def search_data(data_values, coordinates, skip, data_type):
        for d in data_values:
            if d == skip:
                continue
            if data_type == 'float':
                coordinates.append(float(d))
            elif data_type == 'int':
                coordinates.append(int(d)-1)


    @staticmethod # sorted vertex buffer for use with glDrawArrays function
    def create_sorted_vertex_buffer(indices_data, vertices, textures, normals):
        for i, ind in enumerate(indices_data):
            if i % 3 == 0: # sort the vertex coordinates
                start = ind * 3
                end = start + 3
                ObjLoader.buffer.extend(vertices[start:end])
            elif i % 3 == 1: # sort the texture coordinates
                start = ind * 2
                end = start + 2
                ObjLoader.buffer.extend(textures[start:end])
            elif i % 3 == 2: # sort the normal vectors
                start = ind * 3
                end = start + 3
                ObjLoader.buffer.extend(normals[start:end])


    @staticmethod # TODO unsorted vertex buffer for use with glDrawElements function
    def create_unsorted_vertex_buffer(indices_data, vertices, textures, normals):
        num_verts = len(vertices) // 3

        for i1 in range(num_verts):
            start = i1 * 3
            end = start + 3
            ObjLoader.buffer.extend(vertices[start:end])

            for i2, data in enumerate(indices_data):
                if i2 % 3 == 0 and data == i1:
                    start = indices_data[i2 + 1] * 2
                    end = start + 2
                    ObjLoader.buffer.extend(textures[start:end])

                    start = indices_data[i2 + 2] * 3
                    end = start + 3
                    ObjLoader.buffer.extend(normals[start:end])

                    break


    @staticmethod
    def show_buffer_data(buffer):
        for i in range(len(buffer)//8):
            start = i * 8
            end = start + 8
            print(buffer[start:end])


    @staticmethod
    def load_model(file, sorted=True):
        vert_coords = [] # will contain all the vertex coordinates
        tex_coords = [] # will contain all the texture coordinates
        norm_coords = [] # will contain all the vertex normals

        all_indices = [] # will contain all the vertex, texture and normal indices
        indices = [] # will contain the indices for indexed drawing


        with open(file, 'r') as f:
            line = f.readline()
            while line:
                values = line.split()
                if values[0] == 'v':
                    ObjLoader.search_data(values, vert_coords, 'v', 'float')
                elif values[0] == 'vt':
                    ObjLoader.search_data(values, tex_coords, 'vt', 'float')
                elif values[0] == 'vn':
                    ObjLoader.search_data(values, norm_coords, 'vn', 'float')
                elif values[0] == 'f':
                    for value in values[1:]:
                        val = value.split('/')
                        ObjLoader.search_data(val, all_indices, 'f', 'int')
                        indices.append(int(val[0])-1)

                line = f.readline()

        if sorted:
            # use with glDrawArrays
            ObjLoader.create_sorted_vertex_buffer(all_indices, vert_coords, tex_coords, norm_coords)
        else:
            # use with glDrawElements
            ObjLoader.create_unsorted_vertex_buffer(all_indices, vert_coords, tex_coords, norm_coords)

        # ObjLoader.show_buffer_data(ObjLoader.buffer)

        buffer = ObjLoader.buffer.copy() # create a local copy of the buffer list, otherwise it will overwrite the static field buffer
        ObjLoader.buffer = [] # after copy, make sure to set it back to an empty list

        return np.array(indices, dtype='uint32'), np.array(buffer, dtype='float32')

from pyrr import Vector3, vector, vector3, matrix44
from math import sin, cos, radians, atan, sqrt, asin, degrees, acos


class Camera:
    def __init__(self):
        # Позиция камеры
        self.camera_pos = Vector3([0.0, 4.0, 3.0])
        # Рассояние до точки просмотра камеры
        self.camera_front = Vector3([0.0, 0.0, -1.0])
        # Вертикальное перемещение
        self.camera_up = Vector3([0.0, 1.0, 0.0])
        # Горизонтальное перемещение
        self.camera_right = Vector3([1.0, 0.0, 0.0])
        # Чувствительность
        self.mouse_sensitivity = 0.25
        # Вращение вокруг оси y
        self.jaw = -90
        # Вращение вокруг оси x
        self.pitch = 0
        # Индикатор вращения сцены
        self.rotated = False

    def get_view_matrix(self):
        """Матрица расположения камеры"""
        return matrix44.create_look_at(self.camera_pos,self.camera_pos + self.camera_front, self.camera_up)

    def get_view_direction(self):
        """Вектор направления просомтра"""
        def magnifier(num):
            if num<0:
                return num
            else:
                return num
        normal = []
        for i in self.camera_pos + self.camera_front:
            normal.append(magnifier(i))
        return  np.array(normal,dtype = np.float32)

    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        """Изменение углов при перемещении мыши"""
        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity

        self.jaw += xoffset
        self.pitch += yoffset

        if constrain_pitch:
            if self.pitch > 80:
                self.pitch = 80
            if self.pitch < -80:
                self.pitch = -80

        self.update_camera_vectors()

    def update_camera_vectors(self):
        """Изменение матрицы расположения камеры"""
        front = Vector3([0.0, 0.0, 0.0])
        front.x = cos(radians(self.jaw)) * cos(radians(self.pitch))
        front.y = sin(radians(self.pitch))
        front.z = sin(radians(self.jaw)) * cos(radians(self.pitch))

        self.camera_front = vector.normalise(front)
        self.camera_right = vector.normalise(vector3.cross(self.camera_front, Vector3([0.0, 1.0, 0.0])))
        self.camera_up = vector.normalise(vector3.cross(self.camera_right, self.camera_front))

    def process_keyboard(self, direction, velocity):
        """События клавиатуры"""
        if direction == "FORWARD":
            self.camera_pos += self.camera_front * velocity
        if direction == "BACKWARD":
            self.camera_pos -= self.camera_front * velocity
        if direction == "LEFT":
            self.camera_pos -= self.camera_right * velocity
        if direction == "RIGHT":
            self.camera_pos += self.camera_right * velocity
        # Остановка прокрутки сцены
        if direction == "STOP_ROTATE":
            self.rotated = False
        # Прокрутка сцены
        if direction in ["COUNTER_CLOCKWISE","CLOCKWISE"]:
            self.rotated = True
            # Против часовой стрелки
            if direction == "COUNTER_CLOCKWISE":
                camera_position = self.camera_pos
                x,y,z = [camera_position[0],camera_position[1],camera_position[2]]
                # Использование двумерной матрицы поворота
                self.camera_pos = Vector3([x * cos(radians(velocity)) + z * sin(radians(velocity)),
                                            y,
                                            -x * sin(radians(velocity)) + z * cos(radians(velocity))])
            else:
                camera_position = self.camera_pos
                x,y,z = [camera_position[0],camera_position[1],camera_position[2]]
                # Использование двумерной матрицы поворота
                self.camera_pos = Vector3([x * cos(radians(velocity)) - z * sin(radians(velocity)),
                                            y,
                                            x * sin(radians(velocity)) + z * cos(radians(velocity))])
            # Задаем точкой просмотра начало координат
            self.camera_front = -self.camera_pos
            front = vector.normalise(self.camera_front)

            # Восстанавливаем углы поворотов вокруг осей x и z
            self.pitch = degrees(asin((front[1])))
            if self.pitch > 80:
                self.pitch = 80
            if self.pitch < -80:
                self.pitch = -80
            if self.camera_pos[2]>=0:
                self.jaw = degrees(asin(front[2] / cos(radians(self.pitch))))
                if self.camera_pos[2] >= 0 and self.camera_pos[0] >= 0:
                    print('SECOND', self.jaw)
                    self.jaw = -(180+degrees(asin(front[2] / cos(radians(self.pitch)))))
            else:
                self.jaw = degrees(acos(front[0]/ cos(radians(self.pitch))))
            self.update_camera_vectors()


