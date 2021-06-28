import json

import glfw
import pyrr
from OpenGL.GL import *
from OpenGL.GL import glVertexAttribPointer
from OpenGL.GL.shaders import compileProgram, compileShader

from utils import load_texture, ObjLoader, Camera

def read_json(path):
    """Чтение json документа"""
    with open(path, 'r') as f:
        data = json.loads(f.read())
        return data

def position_matrix(coordinates):
    """Матрица перемещения"""
    if not coordinates:
        coordinates = [0,0,0]
    return pyrr.matrix44.create_from_translation(pyrr.Vector3(coordinates))

def scale_matrix(coordinates):
    """Матрица масштаба"""
    if not coordinates:
        coordinates = [1,1,1]
    return pyrr.Matrix44.from_scale(pyrr.Vector3(coordinates))

def rotation_matrix(coefficient, time_loop):
    """Матрица глобального поворота"""
    if not coefficient:
        coefficient = [0,0,0]
    x,y,z = coefficient
    x_rotation = pyrr.Matrix44.from_x_rotation(x*time_loop)
    y_rotation = pyrr.Matrix44.from_y_rotation(y*time_loop)
    z_rotation = pyrr.Matrix44.from_z_rotation(z*time_loop)
    xy_rotation = pyrr.matrix44.multiply(x_rotation,y_rotation)
    return pyrr.matrix44.multiply(xy_rotation,z_rotation)


def model_creating(figures):
    """Создание модели"""

    "Лист можели"
    models_list = []
    for i, figure in enumerate(figures):
        # Загрузка 3D объекта из файла
        indices, buffer = ObjLoader.load_model(f'obj/{figure["name"]}.obj')
        # Массив вершин
        glBindVertexArray(VAO[i])
        # Связать буфер модели
        glBindBuffer(GL_ARRAY_BUFFER, VBO[i])
        # Заполнение буфера
        glBufferData(GL_ARRAY_BUFFER, buffer.nbytes, buffer, GL_STATIC_DRAW)
        # Задание вершин
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, buffer.itemsize * 8, ctypes.c_void_p(0))
        # Задание текстур
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, buffer.itemsize * 8, ctypes.c_void_p(12))
        # Задание нормалей
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, buffer.itemsize * 8, ctypes.c_void_p(20))
        # Наложение текстур
        load_texture(f'textures/{figure["name"]}.jpg', textures[i])
        # Расположение
        position = position_matrix(figure["position"])
        scale = scale_matrix(figure["scale"])
        rotation = figure["rotation"]
        # Объединение матриц в одну (умножается на матрицу поворота при отрисовке)
        model = pyrr.matrix44.multiply(position, scale)
        models_list.append({
            'id': figure["id"],
            'model': model,
            'indices': indices,
            'rotation': rotation,
        })
    return models_list

def draw_model(model_dict):
    """Отрисовка модели"""
    model = model_dict["model"]
    rotation = rotation_matrix(model_dict["rotation"], glfw.get_time())
    #Суммарная матриза расположения, поворота и масштаба
    model = pyrr.matrix44.multiply(model, rotation)

    # Связать массив вершин
    glBindVertexArray(VAO[model_dict["id"]])
    #Связать текстуры
    glBindTexture(GL_TEXTURE_2D, textures[model_dict["id"]])
    #Передать геометрию в шейдер
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    #Отрисовка
    glDrawArrays(GL_TRIANGLES, 0, len(model_dict["indices"]))

#Инициализация камеры
cam = Camera()
WIDTH, HEIGHT = 1280, 720
#Центр экрана
lastX, lastY = WIDTH / 2, HEIGHT / 2
first_mouse = True
#Индикаторы событий
left, right, forward, backward, light,counter_clockwise,clockwise = False, False,False, False, False, False, False

def key_input_clb(window, key, scancode, action, mode):
    """ВВод с клавиатуры"""

    global left, right, forward, backward, light,counter_clockwise,clockwise
    # Закрыть окно
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
    # Вперед
    if key == glfw.KEY_W and action == glfw.PRESS:
        forward = True
    elif key == glfw.KEY_W and action == glfw.RELEASE:
        forward = False
    # Назад
    if key == glfw.KEY_S and action == glfw.PRESS:
        backward = True
    elif key == glfw.KEY_S and action == glfw.RELEASE:
        backward = False
    # Влево
    if key == glfw.KEY_A and action == glfw.PRESS:
        left = True
    elif key == glfw.KEY_A and action == glfw.RELEASE:
        left = False
    # Вправо
    if key == glfw.KEY_D and action == glfw.PRESS:
        right = True
    elif key == glfw.KEY_D and action == glfw.RELEASE:
        right = False
    # Поворот сцены по часовой
    if key == glfw.KEY_E and action == glfw.PRESS:
        clockwise = True
    elif key == glfw.KEY_E and action == glfw.RELEASE:
        clockwise = False
    # Поворот сцены против часовой
    if key == glfw.KEY_Q and action == glfw.PRESS:
        counter_clockwise = True
    elif key == glfw.KEY_Q and action == glfw.RELEASE:
        counter_clockwise = False
    # Фонарик
    if key == glfw.KEY_N and action == glfw.PRESS:
        light = not light


def do_movement():
    """Отслеживание событий в основном цикле"""
    if left:
        cam.process_keyboard("LEFT", 0.15)
    if right:
        cam.process_keyboard("RIGHT", 0.15)
    if forward:
        cam.process_keyboard("FORWARD", 0.15)
    if backward:
        cam.process_keyboard("BACKWARD", 0.15)
    if counter_clockwise:
        cam.process_keyboard("COUNTER_CLOCKWISE", 3)
    if clockwise:
        cam.process_keyboard("CLOCKWISE", 3)
    if not counter_clockwise and not clockwise:
        cam.process_keyboard("STOP_ROTATE", 3)


def mouse_look_clb(window, xpos, ypos):
    """CollBack для мышки """
    global first_mouse, lastX, lastY

    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos

    lastX = xpos
    lastY = ypos

    cam.process_mouse_movement(xoffset, yoffset)


vertex_src = """
# version 330
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
layout(location = 2) in vec3 a_normal;
in vec3 a_lighter_direction;
uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 light_1;
uniform mat4 light_2;
out vec2 v_texture;
out vec3 v_normal_1;
out vec3 v_normal_2;

void main()
{
    v_normal_1 = (light_1 * vec4(a_normal, 0.0f)).xyz;
    v_normal_2 = (light_2 * vec4(a_normal, 0.0f)).xyz;
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_texture = a_texture;
}
"""

fragment_src = """
# version 330
in vec2 v_texture;
in vec3 v_normal_1;
in vec3 v_normal_2;
out vec4 out_color;
uniform sampler2D s_texture;
void main()
{
    vec3 ambientLightIntensity = vec3(0.2f,0.2f,0.2f);
    vec3 lighterLightIntensity = vec3(0.9f,0.9f,0.9f);
    vec3 lighterLightDirection = normalize(vec3(-0.1f,0.0f,0.1f));
    vec3 sunLightIntensity = vec3(0.7f,0.7f,0.7f);
    vec3 sunLightDirection = normalize(vec3(-0.1f,0.0f,-0.1f));
    vec4 texel = texture(s_texture, v_texture);
    vec3 lightIntensity = ambientLightIntensity + sunLightIntensity * max(dot(v_normal_1, sunLightDirection), 0.0f) + lighterLightIntensity * max(dot(v_normal_2, lighterLightDirection), 0.0f);
    out_color = vec4(texel.rgb * lightIntensity,texel.a);
}
"""



def window_resize_clb(window, width, height):
    """CallBack для ресайза"""
    glViewport(0, 0, width, height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 100)
    #Передать перспективу в шейдер
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)


# Инициализауия glfw
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# Создание окна
window = glfw.create_window(WIDTH, HEIGHT, "My OpenGL window", None, None)

# Проверка наличия созданного окна
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# Позиция на экране Windows
glfw.set_window_pos(window, 300, 200)

# Вызов CallBack функции для ресайза
glfw.set_window_size_callback(window, window_resize_clb)
# Вызов CallBack функции мышки
glfw.set_cursor_pos_callback(window, mouse_look_clb)
# Вызов CallBack функции для клавиатуры
glfw.set_key_callback(window, key_input_clb)
# Скрытие курсора
glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

# Выбираем текущее окно
glfw.make_context_current(window)

# Компилируем шейдер на С++
shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

#Получаем фигуры из json файла
figures = read_json('models_composition.json')

#Количество фигур
figures_quantity = len(figures)

# Количество Массивов, текстур и буферов соответствует числу фигур
VAO = glGenVertexArrays(figures_quantity)
VBO = glGenBuffers(figures_quantity)
textures = glGenTextures(figures_quantity)

# Получение итоговых данных для геометрического построения фигур
models_dicts = model_creating(figures)

# Явно указываем использование шейдера
glUseProgram(shader)
# Задний план
glClearColor(0, 0.1,0.1, 0.1)
# Проверка глубины
glEnable(GL_DEPTH_TEST)
# Смешивает вычисленные значения цвета фрагмента со значениями в цветовых буферах.
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
#Задание перспективы
projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, 0.1, 100)

#Привязка к переменным в vertex_src для шейдера
model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")
#Передвигающийся источник света
light_loc_1 = glGetUniformLocation(shader, "light_1")
# Фонарик
light_loc_2 = glGetUniformLocation(shader, "light_2")
# Передать геометрию в шейдер
glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

# Основной цикл приложения
while not glfw.window_should_close(window):
    # Запуск приложения
    glfw.poll_events()
    # Прослушивание событий
    do_movement()

    #Очищаем буферы цвета и глубины
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    #Расположение камеры
    view = cam.get_view_matrix()
    #Направление фонаря
    lighter_direction = cam.get_view_direction()

    #Расположение солнца
    sun_position = position_matrix([-8,9,-8])
    sun_scale = scale_matrix([2,2,2])
    sun_rotation = rotation_matrix([0,-1,0], glfw.get_time())
    sun_model = pyrr.matrix44.multiply(sun_position, sun_scale)
    sun_model = pyrr.matrix44.multiply(sun_model, sun_rotation)

    # Передать расположение камеры в шейдер
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    # Передать расположение солнца в шейдер
    glUniformMatrix4fv(light_loc_1, 1, GL_FALSE,  sun_model)

    #Включение отключение фонаря
    if light:
        glUniformMatrix4fv(light_loc_2, 1, GL_FALSE, view)
    else:
        glUniformMatrix4fv(light_loc_2, 1, GL_TRUE, position_matrix([10,10,10]))

    #Отрисовка фигур
    for model in models_dicts:
        draw_model(model)

    glfw.swap_buffers(window)

# terminate glfw, free up allocated resources
glfw.terminate()
