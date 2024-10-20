#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: Joras Oliveira
Disciplina: Computação Gráfica
Data: 09/08/2024
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy
from typing import Tuple, List, Dict, Optional

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800  # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width * 2
        GL.height = height * 2
        GL.near = near
        GL.far = far
        GL.view_matrix = np.identity(4)
        GL.projection_matrix = np.identity(4)
        GL.transform_matrices = []
        GL.perspective_matrix = np.identity(4)
        GL.vertex_colors = None
        GL.w_values = None
        GL.three_d_call = 0

    @staticmethod
    def barycentric_coordinates(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate barycentric coordinates of point p with respect to triangle abc.
        """
        v0: np.ndarray = b - a
        v1: np.ndarray = c - a
        v2: np.ndarray = p - a
        d00: float = np.dot(v0, v0)
        d01: float = np.dot(v0, v1)
        d11: float = np.dot(v1, v1)
        d20: float = np.dot(v2, v0)
        d21: float = np.dot(v2, v1)
        denom: float = d00 * d11 - d01 * d01

        # Check if the triangle is degenerate
        if abs(denom) < 1e-6:
            return -1.0, -1.0, -1.0  # Return invalid coordinates

        v: float = (d11 * d20 - d01 * d21) / denom
        w: float = (d00 * d21 - d01 * d20) / denom
        u: float = 1.0 - v - w

        return u, v, w

    @staticmethod
    def quarternion_rotation(points: list[int]) -> np.array:
        """Returns the rotation matrix from the quarternion values."""

        #calculatin the quarternions for the rotation
        qi = points[0] * math.sin(points[3] / 2) 
        qj = points[1] * math.sin(points[3] / 2) 
        qk = points[2] * math.sin(points[3] / 2)
        qr = math.cos(points[3] / 2)

        #normalizing the quaternions
        norm = math.sqrt(qi**2 + qj**2 + qk**2 + qr**2)
        qi /= norm
        qj /= norm
        qk /= norm
        qr /= norm

        #building the rotation matrix
        rotation_matrix = np.array([
            [1 - 2*(qj**2 + qk**2), 2*(qi*qj - qk*qr), 2*(qi*qk + qj*qr), 0],
            [2*(qi*qj + qk*qr), 1 - 2*(qi**2 + qk**2), 2*(qj*qk - qi*qr), 0],
            [2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1 - 2*(qi**2 + qj**2), 0],
            [0, 0, 0, 1]
        ]) 
        return rotation_matrix
    @staticmethod
    def is_inside_triangle(x_0, y_0, x_1, y_1, x_2, y_2, x, y):
        l_1 = x * (y_1 - y_0) - y * (x_1 - x_0) + y_0 * (x_1 - x_0) - x_0 * (y_1 - y_0)
        l_2 = x * (y_2 - y_1) - y * (x_2 - x_1) + y_1 * (x_2 - x_1) - x_1 * (y_2 - y_1)
        l_3 = x * (y_0 - y_2) - y * (x_0 - x_2) + y_2 * (x_0 - x_2) - x_2 * (y_0 - y_2)

        return (l_1 >= 0) and (l_2 >= 0) and (l_3 >= 0)



    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Polypoint2D : pontos = {0}".format(point)) # imprime no terminal pontos
        # print("Polypoint2D : colors = {0}".format(colors)) # imprime no terminal as cores

        
        for i in range(0, len(point) - 1, 2): # itera na lista de pontos de 2 em 2 para pegar X e Y de uma vez
            #tudo com type cast p/ int pois a funcao do GPU.draw_pixel apenas recebe ints
            pos_x = int(point[i])
            pos_y = int(point[i + 1])
            r = int(colors["emissiveColor"][0] * 255) #pegamos apenas a cor emissiva, e salvamos cada valor em sua variavel correspondente
            g = int(colors["emissiveColor"][1] * 255)
            b = int(colors["emissiveColor"][2] * 255)
            gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [r, g, b])

        # Exemplo:
        # pos_x = GL.width//2
        # pos_y = GL.height//2
        # gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 0])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        # print("Polyline2D : lineSegments = {0}".format(lineSegments)) # imprime no terminal
        # print("Polyline2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        # Saving the colors in their respective variable, using emissive colors.
        r = int(colors["emissiveColor"][0] * 255) 
        g = int(colors["emissiveColor"][1] * 255)
        b = int(colors["emissiveColor"][2] * 255)
        
        for i in range(0, len(lineSegments), 4):
            x_0 = int(lineSegments[i])
            y_0 = int(lineSegments[i + 1])
            x_1 = int(lineSegments[i + 2])
            y_1 = int(lineSegments[i + 3])

            #bresenhams algorithm, calc based from: https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm
            dx = abs(x_1 - x_0)
            dy = abs(y_1 - y_0)

            sx = -1 if x_0 > x_1 else 1
            sy = -1 if y_0 > y_1 else 1

            gpu.GPU.draw_pixel([x_0, y_0], gpu.GPU.RGB8, [r, g, b])
    
            if dx > dy:
                err = dx / 2.0
                while (x_0 != x_1):
                    gpu.GPU.draw_pixel([x_0, y_0], gpu.GPU.RGB8, [r, g, b])
                    err -= dy
                    if err < 0:
                        y_0 += sy 
                        err += dx 
                    x_0 += sx
            else:             
                err = dy / 2.0
                while (y_0 != y_1):
                    gpu.GPU.draw_pixel([x_0, y_0], gpu.GPU.RGB8, [r, g, b])
                    err -= dx
                    if err < 0:
                        x_0 += sx 
                        err += dy 
                    y_0 += sy

            gpu.GPU.draw_pixel([x_1, y_1], gpu.GPU.RGB8, [r, g, b])

        # Exemplo:
        # GL.polypoint2D(lineSegments, colors)
        # testes para ajudar no debuggin:
        # gpu.GPU.draw_pixel([x_0, y_0], gpu.GPU.RGB8, [r, g, b])
        # gpu.GPU.draw_pixel([x_1, y_1], gpu.GPU.RGB8, [r, g, b]) # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        # print("Circle2D : radius = {0}".format(radius)) # imprime no terminal
        # print("Circle2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        # Exemplo:
        pos_x = GL.width//2
        pos_y = GL.height//2
        gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 255])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)


    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        # Saving the colors in their respective variable, using emissive colors.
        r = int(colors["emissiveColor"][0] * 255) 
        g = int(colors["emissiveColor"][1] * 255)
        b = int(colors["emissiveColor"][2] * 255)

        for i in range(0, len(vertices), 6):
            #all vertices are saved as points
            if GL.three_d_call:
                vertices_2d = vertices[:2, :].flatten(order='F')
                x_0, y_0 = int(vertices_2d[0]), int(vertices_2d[1])
                x_1, y_1 = int(vertices_2d[2]), int(vertices_2d[3])
                x_2, y_2 = int(vertices_2d[4]), int(vertices_2d[5])
                z0, z1, z2 = vertices[2]
            else: 
                x_0, y_0 = int(vertices[0]) * 2, int(vertices[1]) * 2
                x_1, y_1 = int(vertices[2]) * 2, int(vertices[3]) * 2
                x_2, y_2 = int(vertices[4]) * 2, int(vertices[5]) * 2

            # figuring out if a point is withing a triangle or not
            # with bounding box optmization

            # Check if the triangle is completely outside the screen
            if (max(x_0, x_1, x_2) < 0 or min(x_0, x_1, x_2) >= GL.width or
                max(y_0, y_1, y_2) < 0 or min(y_0, y_1, y_2) >= GL.height):
                return  # Skip this triangle

            # Bounding box (clamped to screen dimensions)
            x_min = max(min(x_0, x_1, x_2), 0)
            x_max = min(max(x_0, x_1, x_2), GL.width - 1)
            y_min = max(min(y_0, y_1, y_2), 0)
            y_max = min(max(y_0, y_1, y_2), GL.height - 1)

            v1 = np.array([x_0, y_0])
            v2 = np.array([x_1, y_1])
            v3 = np.array([x_2, y_2])

            # colors for each vertex
            if GL.vertex_colors is not None:
                c1 = np.array(GL.vertex_colors[0:3])
                c2 = np.array(GL.vertex_colors[3:6])
                c3 = np.array(GL.vertex_colors[6:9])
            else:
                c1 = c2 = c3 = np.array(colors["emissiveColor"])

            # Precalculate 1/w for each vertex if w_values are provided
            if GL.w_values is not None:
                w1, w2, w3 = 1/GL.w_values[0], 1/GL.w_values[1], 1/GL.w_values[2]
            else:
                w1 = w2 = w3 = 1.0

            # draws the triangle on the screen and fills it with the color/gradient/texure
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    p = np.array([x, y])
                    is_inside = GL.is_inside_triangle(x_0, y_0, x_1, y_1, x_2, y_2, x, y)
                    if is_inside:
                        u, v, w = GL.barycentric_coordinates(p, v1, v2, v3)
                        z = (u * z0 + v * z1 + w * z2)  / ( ( u*(1/z0) + v*(1/z1) + w*(1/z2) ) )
                        z_buffer = int(gpu.GPU.read_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F))

                        if z < z_buffer:
                            # Perspective-correct interpolation
                            u_corrected = (u * w1) / (u * w1 + v * w2 + w * w3)
                            v_corrected = (v * w2) / (u * w1 + v * w2 + w * w3)
                            w_corrected = (w * w3) / (u * w1 + v * w2 + w * w3)

                            color = u_corrected * c1 + v_corrected * c2 + w_corrected * c3
                            r, g, b = [int(max(0, min(255, c * 255))) for c in color]

                            transparency = colors["transparency"]

                            if (transparency > 0):
                                frame_buffer = gpu.GPU.read_pixel([x, y], gpu.GPU.RGB8)
                                # print("frame_buffer:", frame_buffer)

                                old_r = frame_buffer[0]
                                old_g = frame_buffer[1]
                                old_b = frame_buffer[2] 
                                # Directly multiply and check values
                                r_mult = old_r * transparency
                                g_mult = old_g * transparency
                                b_mult = old_b * transparency
                                r_new = r * (1 - transparency)
                                g_new = g * (1 - transparency)
                                b_new = b * (1 - transparency)
                                r = int(r_new + r_mult)
                                g = int(g_new + g_mult)
                                b = int(b_new + b_mult)

                            gpu.GPU.draw_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F, [z])
                            gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, [r, g, b])
                            
        # Exemplo:
        # GL.polypoint2D(vertices, colors)


    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("TriangleSet : pontos = {0}".format(point)) # imprime no terminal pontos
        # print("TriangleSet : colors = {0}".format(colors)) # imprime no terminal as cores
        GL.three_d_call = 1
        for i in range(0, len(point) - 8, 9):
            # coordiantes array, format is rows contain  x, y, z, w values
            points = np.array([
                [point[i], point[i + 3], point[i + 6]],
                [point[i + 1], point[i + 4], point[i + 7]],
                [point[2], point[i + 5], point[i + 8]],
                [1, 1, 1]
            ])
            # Applying transformations 
            transform = GL.transform_matrices[-1] @ points
            view = GL.view_matrix @ transform
            perspective = GL.perspective_matrix @ view

            # W values for perspective correction later
            w_values = perspective[3, :]
            GL.w_values = w_values

            # Divides all x, y, z values by w to normalize them
            normalized_points = perspective[:3, :] / perspective[3, :]

            # NDC projection
            ndc_projection = np.vstack([
                normalized_points,
                np.ones(normalized_points.shape[1])
            ])

            # Mapping from camera space to screen space
            screen_transform = np.array([
                [GL.width / 2, 0, 0, GL.width / 2],
                [0, GL.height / 2, 0, GL.height / 2],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            projection_matrix = screen_transform @ ndc_projection
        
            GL.triangleSet2D(projection_matrix, colors)


    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # orientation and position matrices
        rotation_matrix = GL.quarternion_rotation(orientation)
        translation_matrix = np.array([
            [1, 0, 0, -position[0]],
            [0, 1, 0, -position[1]],
            [0, 0, 1, -position[2]],
            [0, 0, 0, 1]
        ])

        # look at represents the camera orientation in relation to the world
        # view matrix represents the world orientation in relation to the camera
        look_at = np.linalg.inv(translation_matrix) @ np.linalg.inv(rotation_matrix)
        GL.view_matrix = np.linalg.inv(look_at)

        # fov from camera angles to screen angles
        fov_y = 2 * math.atan(math.tan(fieldOfView / 2) * (GL.height / np.hypot(GL.height, GL.width)))

        # deriving values from constants to be used in perspective matrix
        aspect_ratio = GL.width / GL.height
        top = GL.near * math.tan(fov_y)
        right = top * aspect_ratio

        # Perspective matrix
        perspective_matrix = np.array([
            [(GL.near / right), 0, 0, 0],
            [0, -(GL.near / top), 0, 0],
            [0, 0, -((GL.far + GL.near) / (GL.far - GL.near)), -((2 * GL.far * GL.near) / (GL.far - GL.near))],
            [0, 0, -1, 0]
        ])
        GL.perspective_matrix = perspective_matrix

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        translation_matrix = np.array([
            [1, 0, 0, translation[0]], 
            [0, 1, 0, translation[1]], 
            [0, 0, 1, translation[2]], 
            [0, 0, 0, 1]
        ])
        scale_matrix = np.array([
            [scale[0], 0, 0, 0], 
            [0, scale[1], 0, 0], 
            [0, 0, scale[2], 0], 
            [0, 0, 0, 1]
        ])
        rotation_matrix = GL.quarternion_rotation(rotation)

        # always follow the order: translation -> rotation -> scale
        transformation_matrix = translation_matrix @ rotation_matrix @ scale_matrix
        if GL.transform_matrices:
            GL.transform_matrices.append(GL.transform_matrices[-1] @ transformation_matrix)
        else: GL.transform_matrices.append(transformation_matrix)


    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        if GL.transform_matrices:
            GL.transform_matrices.pop()
        # print("Saindo de Transform")

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("the number of points is: ", len(point))
        # print("the number of points devided by 3 is: ", len(point) / 3)
        # print("TriangleStripSet : pontos = {0} ".format(point), end='')
        # for i, strip in enumerate(stripCount):
        #     print("strip[{0}] = {1} ".format(i, strip), end='')
        # print("")
        # print("TriangleStripSet : colors = {0}".format(colors)) # imprime no terminal as cores
        triangle_strip = []
        offset = 0
        for strip in stripCount:
            for i in range(strip - 2):  # Each strip forms (strip - 2) triangles
                idx1 = offset + i
                idx2 = offset + i + 1
                idx3 = offset + i + 2

                x_0, y_0, z_0 = point[idx1 * 3], point[idx1 * 3 + 1], point[idx1 * 3 + 2]
                x_1, y_1, z_1 = point[idx2 * 3], point[idx2 * 3 + 1], point[idx2 * 3 + 2]
                x_2, y_2, z_2 = point[idx3 * 3], point[idx3 * 3 + 1], point[idx3 * 3 + 2]

                # Change the order for every second triangle to ensure counterclockwise orientation
                points = [(x_0, y_0, z_0), (x_1, y_1, z_1), (x_2, y_2, z_2)]
                if i % 2 != 0:
                    points[1], points[2] = points[2], points[1]

                for point_set in points:
                    triangle_strip.extend(point_set)
            
            offset += strip  # Move to the next set of points

        GL.triangleSet(triangle_strip, colors)


    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("IndexedTriangleStripSet : pontos = {0}, index = {1}".format(point, index))
        # print("IndexedTriangleStripSet : colors = {0}".format(colors)) # imprime as cores

        vertices = {}
        for i in range(0, len(point), 3):
            vertices[i // 3] = (point[i], point[i + 1], point[i + 2])

        triangle_strip = []
        i = 0
        
        while i < len(index):
            # -1 indicates end of current strip
            if index[i] == -1:
                i += 1
                continue

            # Ensure that we do not go out of bounds when fetching index[i+2]
            if i + 2 >= len(index) or index[i + 1] == -1 or index[i + 2] == -1:
                break

            idx1 = index[i]
            idx2 = index[i + 1]
            idx3 = index[i + 2]

            points = [vertices[idx1], vertices[idx2], vertices[idx3]]

            # Adjust for counterclockwise orientation
            if i % 2 != 0:
                points[1], points[2] = points[2], points[1]

            for point_set in points:
                triangle_strip.extend(point_set)

            i += 1 

        GL.triangleSet(triangle_strip, colors)


    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.


        #creagin a list of vertices and colors
        vertices = {}
        for i in range(0, len(coord), 3):
            vertices[i // 3] = coord[i:i+3]

        vertex_colors = {}
        if colorPerVertex and color:
            for i in range(0, len(color), 3):
                vertex_colors[i // 3] = (color[i:i+3])

        tex_coords = {}
        if texCoord and texCoordIndex:
            for i in range(0, len(texCoord), 3):
                tex_coords[i // 3] = texCoord[i:i+3]

        #calculating the triangles 
        i = 0
        while i < len(coordIndex):
            if coordIndex[i] == -1:
                i += 1
                continue

            face_indices = []
            while i < len(coordIndex) and coordIndex[i] != -1:
                face_indices.append(coordIndex[i])
                i += 1
 
            for j in range(1, len(face_indices) - 1):
                idx1, idx2, idx3 = face_indices[0], face_indices[j], face_indices[j + 1]
                v1, v2, v3 = vertices[idx1], vertices[idx2], vertices[idx3]

                triangle_points = v1 + v2 + v3

                if colorPerVertex and color:
                    c1, c2, c3 = vertex_colors[idx1], vertex_colors[idx2], vertex_colors[idx3]
                    triangle_colors = c1 + c2 + c3
                    GL.vertex_colors = triangle_colors
                else:
                    triangle_colors = colors["emissiveColor"] * 3

                GL.triangleSet(triangle_points, colors)

            i += 1
            



    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
