# 이것은 저의 다른 게임 프로젝트입니다.
# 엔진 개발할때 만든 템플릿이라 매우 완벽한 샘플이기 때문에 example 로 사용하도록 합니다.

from kawaraPGE import kwrpge
import pygame

# Add Game and Scene
game = kwrpge.Game()
game.fill_color = pygame.Color(53, 55, 59)
pcb_scene = kwrpge.Scene("PCB")
game.add_scene(pcb_scene)

# Add Scene to Camera
camera_index = game.scenes[game.get_scene_index_by_name("PCB")].register_object(kwrpge.Camera())
game.scenes[game.get_scene_index_by_name("PCB")].set_camera_by_index(camera_index)

# Sprite Regist
pins = kwrpge.ShapeSprite(kwrpge.ANTI_CIRCLE, pygame.Color(101, 107, 117), kwrpge.radius_to_vector2(10))
pin_obj = kwrpge.SpriteObject(pygame.Vector2(-2,-2),pins,"Pins",kwrpge.Pivot.TOP_LEFT,pygame.Vector2(200,100))
# print("test Pivot Offset:",pin_obj.get_pivot_offset())
# print("test Pivot Pos:",pin_obj.get_pivot_pos())

# Set Func
def event_handler(event: pygame.Event):
    pass

def loop_func(dt):
    # pin_obj.draw_pivot(screen=game.screen)
    pass

game.set_event_handler(event_handler)
game.set_loop_func(loop_func)

for x in range(0, 1):
    for y in range(0, 1):
        pin_obj.add_copy_position(pygame.Vector2(x*25,y*25))

pcb_scene.register_object(pin_obj)

game.run()