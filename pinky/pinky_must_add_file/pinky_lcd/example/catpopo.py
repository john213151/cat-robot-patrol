from pinky_lcd import LCD
from PIL import Image, ImageSequence
import time

lcd = LCD()

print("Start GIF")

img = Image.open("catpopo.gif")

start_time = time.time()

# 현재 시간 - 시작 시간이 10보다 작은 동안 무한 반복
#while(1):
while time.time() - start_time < 300:
    for frame in ImageSequence.Iterator(img):
        time.sleep(1/24)
        lcd.img_show(frame)

#
    

print("End GIF")

lcd.close()


