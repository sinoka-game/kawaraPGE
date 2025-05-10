# ---[ KawaraPGE v1.0 ]---
# Kanowara Python Game Engine
# Copyright (c) 2025, Sinoka Games. All rights reserved.
# Created by Kanowara Sinoka (Kim Taeyang)
#
# Licensed under the BSD 3-Clause License.
# See the LICENSE file distributed with this source code for details.
#
# To use this engine, place the "kwrpge" folder in your project directory,
# and import it with the following command:
# 
# import kwrpge
#
# --- [ All Code ] ---

import pygame
from typing import List
from enum import Enum, auto

class Pivot(Enum):
    TOP_LEFT = auto()
    TOP = auto()
    TOP_RIGHT = auto()
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM = auto()
    BOTTOM_RIGHT = auto()

class ObjectType:
    def __init__(
        self,
        pos: pygame.Vector2,
        name: str,
        pivot: Pivot = Pivot.TOP_LEFT,
        size: pygame.Vector2 = pygame.Vector2(32, 32),
        z_index: int = 0
    ):
        self.pos = pos
        self.name = name
        self.pivot = pivot
        self.size = size
        self.z_index = z_index

    def get_pivot_offset(self) -> pygame.Vector2:
        center_x = self.size.x / 2
        center_y = self.size.y / 2
        x = self.size.x
        y = self.size.y
        zero = 0

        match self.pivot:
            case Pivot.TOP_LEFT:
                return pygame.Vector2(zero, zero)
            case Pivot.TOP:
                return pygame.Vector2(center_x, zero)
            case Pivot.TOP_RIGHT:
                return pygame.Vector2(x, zero)
            case Pivot.LEFT:
                return pygame.Vector2(zero, center_y)
            case Pivot.CENTER:
                return pygame.Vector2(center_x, center_y)
            case Pivot.RIGHT:
                return pygame.Vector2(x, center_y)
            case Pivot.BOTTOM_LEFT:
                return pygame.Vector2(zero, y)
            case Pivot.BOTTOM:
                return pygame.Vector2(center_x, y)
            case Pivot.BOTTOM_RIGHT:
                return pygame.Vector2(x, y)

    def get_pivot_pos(self):
        return self.pos - self.get_pivot_offset()

    def update(self, dt: float):
        pass  # 기본 동작 없음

class Sprite:
    def __init__(self, image: pygame.Surface):
        self.image = image

    def draw(self, screen: pygame.Surface, pos: pygame.Vector2):
        screen.blit(self.image, pos)

class ShapeSprite(Sprite):
    def __init__(self, shape_type: str, color: pygame.Color, size: pygame.Vector2, thickness: int = 0):
        """
        Create a shape-based sprite.

        :param shape_type: The type of shape ('circle', 'rectangle', etc.)
        :param color: The color of the shape (pygame.Color)
        :param size: The size of the shape (pygame.Vector2). For a circle, this is the radius; for a rectangle, it's the width and height.
        :param thickness: The thickness of the border (0 for filled shapes).
        """
        self.shape_type = shape_type
        self.color = color
        self.size = size
        self.thickness = thickness
        
        # Create a surface to draw the shape on
        self.image = pygame.Surface(size, pygame.SRCALPHA)  # Use SRCALPHA for transparency
        self._draw_shape()

    def _draw_shape(self):
        """Draw the selected shape onto the surface."""
        if self.shape_type == 'circle':
            # Draw a circle (size.x is radius)
            pygame.draw.circle(self.image, self.color, (self.size.x // 2, self.size.y // 2), self.size.x // 2, self.thickness)
        elif self.shape_type == 'rectangle':
            # Draw a rectangle (size.x is width, size.y is height)
            pygame.draw.rect(self.image, self.color, pygame.Rect(0, 0, self.size.x, self.size.y), self.thickness)
        elif self.shape_type == 'triangle':
            # Draw a triangle (equilateral, centered)
            points = [(self.size.x // 2, 0), (0, self.size.y), (self.size.x, self.size.y)]
            pygame.draw.polygon(self.image, self.color, points, self.thickness)

    def draw(self, screen: pygame.Surface, pos: pygame.Vector2):
        """Draw the shape sprite to the screen at the given position."""
        super().draw(screen, pos)

class SpriteObject(ObjectType):
    def __init__(self, pos: pygame.Vector2, sprite: Sprite, name: str = "Sprite", pivot: Pivot = Pivot.TOP_LEFT, size: pygame.Vector2 = pygame.Vector2(32, 32)):
        super().__init__(pos, name, pivot, size)
        self.sprite = sprite

class Camera(ObjectType):
    def __init__(self, pos: pygame.Vector2 = pygame.Vector2(0, 0), name: str = "Camera", pivot: Pivot = Pivot.TOP_LEFT, size: pygame.Vector2 = pygame.Vector2(800, 600)):
        super().__init__(pos, name, pivot, size)

    def real_pos_to_render_pos(self, real_pos: pygame.Vector2):
        return real_pos - (self.pos - self.get_pivot_offset())

class Scene:
    def __init__(self, name: str):
        self.name = name
        self.objects: list[ObjectType] = []
        self.camera_index = -1

    def register_object(self, add_object: ObjectType):
        self.objects.append(add_object)
        return len(self.objects) - 1

    def set_camera_by_index(self, index: int):
        if isinstance(self.objects[index], Camera):
            self.camera_index = index
        else:
            raise TypeError("Selected object is not a Camera.")

    def get_camera(self):
        if self.camera_index != -1:
            return self.objects[self.camera_index]
        return None

    def update(self, dt: float):
        for obj in self.objects:
            obj.update(dt)

    def draw_objects(self, screen: pygame.Surface):
        if self.camera_index != -1:
            camera = self.objects[self.camera_index]
            for obj in sorted(self.objects, key=lambda o: o.z_index):
                if isinstance(obj, SpriteObject):
                    pos = camera.real_pos_to_render_pos(obj.get_pivot_pos())
                    obj.sprite.draw(screen, pos)


class Game:
    def __init__(self, screen_size=(800, 600)):
        pygame.init()
        self.scenes: dict[str, Scene] = {}
        self.current_scene_name: str | None = None
        self.screen = pygame.display.set_mode(screen_size)
        self.clock = pygame.time.Clock()
        self.event_handler = None
        self.loop_func = None
        self.is_running = False

    def add_scene(self, scene: Scene):
        self.scenes[scene.name] = scene
        if self.current_scene_name is None:
            self.current_scene_name = scene.name

    def set_scene(self, scene_name: str):
        if scene_name in self.scenes:
            self.current_scene_name = scene_name
        else:
            raise ValueError(f"Scene '{scene_name}' not found.")

    def set_event_handler(self, handler):
        self.event_handler = handler

    def set_loop_func(self, func):
        self.loop_func = func

    def get_current_scene(self) -> Scene:
        return self.scenes[self.current_scene_name]

    def run(self):
        self.is_running = True
        while self.is_running:
            # 이벤트 처리
            if self.event_handler:
                for event in pygame.event.get():
                    self.event_handler(event)

            dt = self.clock.tick(60) / 1000.0

            # 현재 씬 업데이트
            current_scene = self.get_current_scene()
            current_scene.update(dt)

            # 사용자 정의 루프 함수
            if self.loop_func:
                self.loop_func()

            # 렌더링
            self.screen.fill((0, 0, 0))
            current_scene.draw_objects(self.screen)
            pygame.display.flip()