import cv2
import numpy as np
import os
from collections import deque

AntalKroner = 0
Threshold = 0.8
FolderDir = r"C:/Users/fifik/OneDrive/Skrivebord/BilledBehandling/CroppedBoards" 
ImagePath = os.path.join(FolderDir, "1.jpg")
InputImg = cv2.imread(ImagePath)

if InputImg is None:
    print("Billede er ikke loaded")
    exit()

FeltType = np.zeros((5, 5))  #Array til felt typer da spillepladen er 5x5
HSVIMG = cv2.cvtColor(InputImg, cv2.COLOR_BGR2HSV)

#HSV Threshold fundet gennem ImageJ HOpenCV = HImageJ*180/255
GrasMin = np.array([33, 189, 123])
GrasMax = np.array([64, 255, 186])

MarkMin = np.array([61, 0, 130])
MarkMax = np.array([102, 255, 255])

VandMin = np.array([102, 0, 0])
VandMax = np.array([107, 255, 255])

SkovMin = np.array([35, 39, 0])
SkovMax = np.array([69, 255, 102])

MoseMin = np.array([16, 0, 12])
MoseMax = np.array([28, 146, 205])

MineMin = np.array([1, 0, 0])
MineMax = np.array([81, 51, 28])

# 3x3 kernel bruges til morphology
kernel = np.ones((5, 5), np.uint8)


def morphological_process(mask):
    """Vi bruger kernel morphology
    til at filtere masken"""
    #Fjerner noise
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Fylder små huller
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    return mask_closed


def BestemFelt(TileHSV): 
    GrasMask = cv2.inRange(TileHSV, GrasMin, GrasMax)
    GrasMask = morphological_process(GrasMask)

    MarkMask = cv2.inRange(TileHSV, MarkMin, MarkMax)
    MarkMask = morphological_process(MarkMask)

    VandMask = cv2.inRange(TileHSV, VandMin, VandMax)
    VandMask = morphological_process(VandMask)

    SkovMask = cv2.inRange(TileHSV, SkovMin, SkovMax)
    SkovMask = morphological_process(SkovMask)

    MoseMask = cv2.inRange(TileHSV, MoseMin, MoseMax)
    MoseMask = morphological_process(MoseMask)

    MineMask = cv2.inRange(TileHSV, MineMin, MineMax)
    MineMask = morphological_process(MineMask)
    
    
    MeanMask = [np.mean(GrasMask), np.mean(MarkMask),np.mean(VandMask),np.mean(SkovMask),np.mean(MoseMask), np.mean(MineMask)]
    # Vi tjekker i hvilket biom passer flest pixels med i forhold til HSV threshold
    if max(MeanMask)>Threshold:
        if np.mean(GrasMask) == max(MeanMask):
            return 1  # Gras
        elif np.mean(MarkMask) == max(MeanMask):
            return 2  # Mark
        elif np.mean(VandMask) == max(MeanMask):
            return 3  # Vand
        elif np.mean(SkovMask) == max(MeanMask):
            return 4  # Skov
        elif np.mean(MoseMask) == max(MeanMask):
            return 5  # mose
        elif np.mean(MineMask) == max(MeanMask):
            return 6  # Mine
    else:
        return 0  # Ingen match 


# Loop gennem billedet og klasificer vært felt
height, width = InputImg.shape[:2]
if height == 500 and width == 500:
    FeltStørelse = 100
    for row in range(0, height, FeltStørelse):
        for col in range(0, width, FeltStørelse):
            TileHSV = HSVIMG[row:row+FeltStørelse, col:col+FeltStørelse]
            field_type = BestemFelt(TileHSV)
            FeltType[row // FeltStørelse, col // FeltStørelse] = field_type


# Grassfire (Flood Fill) Algorithm
def grassfire(matrix):
    rows, cols = matrix.shape
    region_label = 1  # Start region labels from 1
    regions = np.zeros_like(matrix, dtype=int)  # To store labeled regions
    visited = np.zeros_like(matrix, dtype=bool)

    # Directions for N, S, E, W (4-connectivity)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def flood_fill(r, c, current_label, field_type):
        # Use a queue for breadth-first search (BFS)
        queue = deque([(r, c)])
        visited[r, c] = True
        regions[r, c] = current_label
        
        while queue:
            x, y = queue.popleft()

            # Explore in all directions
            for dr, dc in directions:
                nx, ny = x + dr, y + dc

                # Check bounds and if the tile is unvisited and of the same type
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                    if matrix[nx, ny] == field_type:
                        visited[nx, ny] = True
                        regions[nx, ny] = current_label
                        queue.append((nx, ny))

    for r in range(rows):
        for c in range(cols):
            if not visited[r, c] and matrix[r, c] > 0:
                # If the field type is not 0, start a flood fill
                flood_fill(r, c, region_label, matrix[r, c])
                region_label += 1

    return regions

# Apply Grassfire to the field type matrix
labeled_regions = grassfire(FeltType)
print("Labeled regions:")
print(labeled_regions)

# Krone Template Matching
ImgKrone = cv2.imread("krone.png")
ImgKroneRoteret90 = cv2.rotate(ImgKrone, cv2.ROTATE_90_CLOCKWISE)
ImgKroneRoteret180 = cv2.rotate(ImgKroneRoteret90, cv2.ROTATE_90_CLOCKWISE)
ImgKroneRoteret270 = cv2.rotate(ImgKroneRoteret180, cv2.ROTATE_90_CLOCKWISE)

GraaBillede = cv2.cvtColor(InputImg, cv2.COLOR_BGR2GRAY)
GraaKrone = cv2.cvtColor(ImgKrone, cv2.COLOR_BGR2GRAY)
GraaKrone90 = cv2.cvtColor(ImgKroneRoteret90, cv2.COLOR_BGR2GRAY)
GraaKrone180 = cv2.cvtColor(ImgKroneRoteret180, cv2.COLOR_BGR2GRAY)
GraaKrone270 = cv2.cvtColor(ImgKroneRoteret270, cv2.COLOR_BGR2GRAY)


def TemplateMatch(Templaten, GraaBillede, TemplatensNavn):
    global AntalKroner
    Result = cv2.matchTemplate(GraaBillede, Templaten, cv2.TM_CCOEFF_NORMED)
    Lokation = np.where(Result >= Threshold)

    for pt in zip(*Lokation[::-1]):
        AntalKroner += 1
        cv2.rectangle(InputImg, pt, (pt[0] + Templaten.shape[1], pt[1] + Templaten.shape[0]), (0, 0, 255), 2)
        print(f"Match fundet til template {TemplatensNavn} på position x: {pt}")
""" Den her funktion er til at printe og se hvad de hvordan de forskelige threshold ser ud
def show_grass_mask(HSVIMG):
    GrasMask = cv2.inRange(HSVIMG, MoseMin, MoseMax)

    # Show the Grass mask
    cv2.imshow('Gras Mask', GrasMask)
    cv2.waitKey(0)  # Wait for key press to close window
    cv2.destroyAllWindows()  # Close the window after key press

# Convert input image to HSV
HSVIMG = cv2.cvtColor(InputImg, cv2.COLOR_BGR2HSV)

# Call the function to display the grass mask
show_grass_mask(HSVIMG)
"""
TemplateMatch(GraaKrone, GraaBillede, "OriginalKrone")
TemplateMatch(GraaKrone90, GraaBillede, "OriginalKrone")
TemplateMatch(GraaKrone180, GraaBillede, "OriginalKrone")
TemplateMatch(GraaKrone270, GraaBillede, "OriginalKrone")
print("Antal Kroner på billedet", AntalKroner)
print("Typer af felter", FeltType)
print("Labeled regions matrix", labeled_regions)

cv2.imshow('Originalt Billede', InputImg)
cv2.imshow('Krone', ImgKrone)
cv2.imshow('Krone90', ImgKroneRoteret90)
cv2.imshow('Krone180', ImgKroneRoteret180)
cv2.imshow('Krone270', ImgKroneRoteret270)
cv2.waitKey(0)
cv2.destroyAllWindows()
