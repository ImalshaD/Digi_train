import OcrToTableTool as ottt
import TableExtractor as te
import TableLinesRemover as tlr
import cv2

path_to_image = "image\\0.png"
table_extractor = te.TableExtractor(path_to_image)
perspective_corrected_image = table_extractor.execute()



lines_remover = tlr.TableLinesRemover(perspective_corrected_image)
image_without_lines = lines_remover.execute()


ocr_tool = ottt.OcrToTableTool(image_without_lines, perspective_corrected_image)
ocr_tool.execute()