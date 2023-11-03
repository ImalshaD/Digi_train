from img2table.document import Image
import cv2
# Instantiation of the image
img = Image(src="index.png")
cv2img = cv2.imread('index.png')

# Table identification
img_tables = img.extract_tables()

# Result of table identification
ex=img_tables[0]
for i in ex.content:
    for cell in ex.content[i]:
        x=cell.bbox
        x1,y1,x2,y2=x.x1,x.y1,x.x2,x.y2
        cv2.rectangle(cv2img, (x1, y1), (x2, y2), (0,255,0), 1)
cv2.imwrite("test.png",cv2img)