# ---[ KawaraPGE v1.2 ]---
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
# from kawaraPGE import kwrpge
#
# --- [ All Code ] ---

import pygame
from typing import List
from enum import Enum, auto

from ..log4py.log4py import Logger

CIRCLE = 100
RECTANGLE = 101
TRIANGLE = 102

COLOR_BLACK = pygame.Color(0,0,0)

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

# OBJ Copy
# 카와라PGE의 오브젝트 복제 시스템.
# pos는 기본 위치 리스트이며, 각 위치를 기준으로 objcopy_pos에 지정된 상대 오프셋만큼 복제됩니다.
# 모든 그려지는 위치는 공통된 pivot 기준으로 보정되어 렌더링됩니다.

class ObjectType:
    def __init__(
        self,
        pos: pygame.Vector2,
        name: str,
        pivot: Pivot = Pivot.TOP_LEFT,
        size: pygame.Vector2 = pygame.Vector2(32, 32),
        z_index: int = 0,
        objcopy_pos: List[pygame.Vector2] = None  # 오브젝트 복제 위치 리스트
    ):
        self.pos = pos
        self.name = name
        self.pivot = pivot
        self.size = size
        self.z_index = z_index
        self.objcopy_pos = objcopy_pos or []  # 빈 리스트로 초기화 (복제 없음)

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
        
    def get_all_positions(self) -> List[pygame.Vector2]:
        """
        기본 위치와 복제 위치(기본 위치 + 상대 오프셋)의 리스트를 반환합니다.
        """
        positions = [self.pos]  # 기본 위치 포함
        
        # 복제된 모든 위치 추가
        for offset in self.objcopy_pos:
            positions.append(self.pos + offset)
            
        return positions
        
    def get_all_pivot_positions(self) -> List[pygame.Vector2]:
        """
        모든 위치(기본 위치 및 복제 위치)에 대한 pivot 보정된 위치 목록을 반환합니다.
        """
        pivot_offset = self.get_pivot_offset()
        return [pos - pivot_offset for pos in self.get_all_positions()]

    def update(self, dt: float):
        pass  # 기본 동작 없음
        
    def add_copy_position(self, offset: pygame.Vector2):
        """
        복제 위치 추가하기
        """
        self.objcopy_pos.append(offset)
        
    def set_copy_positions(self, offsets: List[pygame.Vector2]):
        """
        복제 위치 목록 설정하기
        """
        self.objcopy_pos = offsets.copy()  # 목록 복사하여 할당
        
    def clear_copy_positions(self):
        """
        모든 복제 위치 제거하기
        """
        self.objcopy_pos.clear()

class Sprite:
    def __init__(self, image: pygame.Surface):
        self.image = image

    def draw(self, screen: pygame.Surface, pos: pygame.Vector2):
        screen.blit(self.image, pos)

class ShapeSprite(Sprite):
    def __init__(self, shape_type: int, color: pygame.Color, size: pygame.Vector2, thickness: int = 0):
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
        if self.shape_type == 100:
            # Draw a circle (size.x is radius)
            pygame.draw.circle(self.image, self.color, (self.size.x // 2, self.size.y // 2), self.size.x // 2, self.thickness)
        elif self.shape_type == 101:
            # Draw a rectangle (size.x is width, size.y is height)
            pygame.draw.rect(self.image, self.color, pygame.Rect(0, 0, self.size.x, self.size.y), self.thickness)
        elif self.shape_type == 102:
            # Draw a triangle (equilateral, centered)
            points = [(self.size.x // 2, 0), (0, self.size.y), (self.size.x, self.size.y)]
            pygame.draw.polygon(self.image, self.color, points, self.thickness)

    def draw(self, screen: pygame.Surface, pos: pygame.Vector2):
        """Draw the shape sprite to the screen at the given position."""
        super().draw(screen, pos)

class SpriteObject(ObjectType):
    def __init__(self, pos: pygame.Vector2, sprite: Sprite, name: str = "Sprite", pivot: Pivot = Pivot.TOP_LEFT, 
                 size: pygame.Vector2 = pygame.Vector2(32, 32), objcopy_pos: List[pygame.Vector2] = None):
        super().__init__(pos, name, pivot, size, 0, objcopy_pos)
        self.sprite = sprite

class Camera(ObjectType):
    def __init__(self, pos: pygame.Vector2 = pygame.Vector2(0, 0), name: str = "Camera", pivot: Pivot = Pivot.TOP_LEFT, 
                 size: pygame.Vector2 = pygame.Vector2(800, 600)):
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
        if isinstance(add_object, Camera):
            self.set_camera_by_index(len(self.objects) - 1)
            
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
                    # 기본 위치와 모든 복제 위치에 대해 그리기
                    for pos in obj.get_all_pivot_positions():
                        render_pos = camera.real_pos_to_render_pos(pos)
                        obj.sprite.draw(screen, render_pos)

    # 씬 내에서 오브젝트의 이름으로 찾기
    def get_object_by_name(self, name: str):
        for obj in self.objects:
            if hasattr(obj, 'name') and obj.name == name:
                return obj
        return None

    # 씬 내에서 오브젝트의 타입으로 찾기
    def get_objects_by_type(self, obj_type: type):
        return [obj for obj in self.objects if isinstance(obj, obj_type)]


class Game:
    def __init__(self, screen_size=(800, 600)):
        self.logger = Logger()
        pygame.init()
        self.scenes: list[Scene] = []
        self.scene_names: dict[str, int] = {}  # 씬 이름과 인덱스를 매핑하는 딕셔너리
        self.current_scene_index: int | None = None  # 현재 씬의 인덱스
        self.screen = pygame.display.set_mode(screen_size)
        self.clock = pygame.time.Clock()
        self.event_handler = None
        self.loop_func = None
        self.is_running = False

    def add_scene(self, scene: Scene):
        # 씬을 리스트에 추가하고 이름-인덱스를 매핑
        self.scenes.append(scene)
        self.scene_names[scene.name] = len(self.scenes) - 1
        if self.current_scene_index is None:
            # 첫 번째 씬을 현재 씬으로 설정
            self.current_scene_index = len(self.scenes) - 1

    def set_scene(self, scene_index: int):
        if 0 <= scene_index < len(self.scenes):
            self.current_scene_index = scene_index
        else:
            raise ValueError(f"Scene at index '{scene_index}' not found.")
    
    # 씬 이름으로 인덱스 찾기
    def get_scene_index_by_name(self, scene_name: str) -> int:
        if scene_name in self.scene_names:
            return self.scene_names[scene_name]
        else:
            raise ValueError(f"Scene with name '{scene_name}' not found.")
    
    def get_current_scene(self) -> Scene:
        if self.current_scene_index is not None:
            return self.scenes[self.current_scene_index]
        raise ValueError("No current scene set.")

    def set_event_handler(self, handler):
        self.event_handler = handler

    def set_loop_func(self, func):
        self.loop_func = func

    def run(self):
        self.is_running = True
        while self.is_running:
            # 이벤트 처리
            if self.event_handler:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.is_running = False
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
        pygame.quit()