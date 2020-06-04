import numpy as np
import fractions
#initialize the pygame
import pygame, sys,os
import math
#initialize the pygame
pygame.init()


def blitRotate(surf, image, pos1, originPos, angle):
    # calcaulate the axis aligned bounding box of the rotated image
        w, h       = image.get_size()
        box        = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box    = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box    = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

    # calculate the translation of the pivot
        pivot        = pygame.math.Vector2(originPos[0], -originPos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move   = pivot_rotate - pivot
# calculate the upper left origin of the rotated image
        origin = (pos[0] - originPos[0] + min_box[0] - pivot_move[0], pos[1] - originPos[1] - max_box[1] + pivot_move[1])
        return origin
    # # get a rotated image
    #     rotated_image = pygame.transform.rotate(image, angle)
    #
    # # rotate and blit the image
    #     surf.blit(rotated_image, origin)
    #
    # # draw rectangle around the image
    #     pygame.draw.rect (surf, (255, 0, 0), (*origin, *rotated_image.get_size()),2)

BACKGROUND_COLOR = (50,50,50)
WIDTH = 800
HEIGHT = 600
w=50
h=50
pressed = False
x=300
y=400
pos =(x,y)
speed = 1
moving = True
angle = 30
screen = pygame.display.set_mode((WIDTH, HEIGHT))
image = pygame.image.load('wheelchair.png')
image = pygame.transform.scale(image, (w, h))
Xdirec =0
Ydirec =1
theta = 0
turntheta = 0
turnangle = 30
clock = pygame.time.Clock()
count = 0
flag = True

while flag:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            sys.exit()

        elif event.type == pygame.KEYDOWN:
            pressed = True

            if event.key == pygame.K_LEFT:
                theta = theta - turnangle
                Xdirec = math.sin(math.radians(theta)) * speed
                Ydirec = math.cos(math.radians(theta)) * speed
                moving = True
                turntheta = -theta

            #x component needs to change as well as y component

            if event.key == pygame.K_RIGHT:
                theta = theta + turnangle
                Xdirec = math.sin(math.radians(theta)) * speed
                Ydirec = math.cos(math.radians(theta)) * speed
                moving = True
                turntheta = -theta



            if event.key == pygame.K_SPACE and moving == True:
                moving = False

            #x component needs to change according to moving

            #if event.key == pygame.K_SPACE and moving == False:
                #moving = True
            #else:
                #moving = False
    screen.fill(BACKGROUND_COLOR)

    clock.tick(30)

    if moving:
        x += Xdirec * speed
        y -= Ydirec * speed
        pos = (x, y)

    origin = blitRotate(screen, image, pos, (w // 2, h // 2), turntheta)
    # get a rotated image
    rotated_image = pygame.transform.rotate(image, turntheta)

    # rotate and blit the image
    screen.blit(rotated_image, origin)

    # draw rectangle around the image
    pygame.draw.rect(screen, (255, 0, 0), (*origin, *rotated_image.get_size()), 2)



    pygame.display.update()



