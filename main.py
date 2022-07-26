import os
import re
import string
import pydicom
import numpy as np
from matplotlib import pyplot
import mpl_toolkits.mplot3d
from pprint import pprint
import time
from stl import mesh

PathDicom = 'images'
PathSTL = 'stl-file/seg.stl'

def imageAnalysis():
    lstFilesDCM = []
    # Прочитать все файлы dicom
    for diName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if '.dcm' and '-0002-' in filename.lower():
                lstFilesDCM.insert(0, os.path.join(diName, filename))

    RefDs = pydicom.read_file(lstFilesDCM[0])
    # print(RefDs)
    # print(RefDs.pixel_array)
    # print(RefDs.PatientPosition)
    # pyplot.imshow(RefDs.pixel_array, cmap=pyplot.cm.bone)
    # pyplot.show() 
    # Отображение осевой поверхности
    # pyplot.axes().set_aspect('equal', 'datalim')
    # pyplot.axes().set_aspect('equal', 'datalim')
    # Меняем картинку на серый цвет
    # pyplot.set_cmap(pyplot.gray())

    # pyplot.imshow (ArrayDicom[:,:, -1]) # Третье измерение указывает, какой слой в данный момент отображается 
    # pyplot.imshow(ArrayDicom[:, 90, :])
    # pyplot.show() 
    # pyplot.figure(dpi=100) 
    # pyplot.axes().set_aspect('equal', 'datalim') 
    # pyplot.set_cmap(pyplot.gray()) 
    # pyplot.imshow(ArrayDicom[:, 90, :])
    # pyplot.show()

    # Исходный снимок
    # data = np.load('seg-files/segmentation.npz')
    # seg = data['arr_0']

    # Маска по органу
    data = np.load('seg-files/segmentationseg.npz')
    seg_mask = data['arr_0']
    ConstPixelDims = (len(lstFilesDCM), int(RefDs.Rows), int(RefDs.Columns))
    print(f'Количество слайсов: {ConstPixelDims[0]}\nРазмер снимка (ширина x высота): '
        f'{ConstPixelDims[1]} x {ConstPixelDims[2]} px.')
    
    # Получить значение интервала
    # PixelSpacing длина и ширина каждого пикселя, единицы (мм)
    # SliceThickness толщина каждого слайса, единица (мм)
    ConstPixelSpacing = (float(RefDs.SliceThickness), float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]))
    print(f'Расстояние между слайсами: {ConstPixelSpacing[0]} мм.')
    print(f'Размер пикселя: {ConstPixelSpacing[1]} мм. x {ConstPixelSpacing[2]} мм.')

    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # Обойти все файлы dicom, прочитать данные изображения и сохранить их в массиве numpy
    for filenameDCM in lstFilesDCM:
        ds = pydicom.read_file(filenameDCM)
        ArrayDicom[lstFilesDCM.index(filenameDCM), :, : ] = ds.pixel_array

    # Наложение маски
    # for i in range(len(ArrayDicom)):
    #     for j in range(len(ArrayDicom[i])):
    #         for k in range(len(ArrayDicom[i][j])):
    #             ArrayDicom[i][j][k] *= seg_mask[i][j][k]
    dicomMask = ArrayDicom * seg_mask     
    dicomMask -= 1000

    # Среднее значение плотности органа
    meanDensity = np.mean(dicomMask[dicomMask > -500])
    print(f'Среднее значение плотности печени = {meanDensity} HU')

    # Значение медианы плотности органа
    medianDensity = np.median(dicomMask[dicomMask > -500])
    print(f'Значение медианы плотности печени = {medianDensity} HU')

    # Среднеквадратичное отклонение
    standardDeviation = np.std(dicomMask[dicomMask > -500])
    print(f'Значение среднеквадратичного отклонения по плотности печени = {standardDeviation} HU')

    # Вычисление максимального размера по осям X и Y
    def organ_size(mask: np.array, axis: string):
        startTime = time.time()
        axis = axis.lower()
        size = 0
        for i in range(len(mask)):
            temp_sum = 0
            if np.sum(mask[i] > 0):
                transformedMask = mask[i].transpose() if axis == 'x' else mask[i] 
                for row in (transformedMask):
                    temp_sum += 1 if np.sum(row > 0) else 0
            size = temp_sum if temp_sum > size else size
        time_calc = np.round(time.time() - startTime, 3)
        return size, time_calc

    # Размер органа по оси X
    sizeX, calcTime = organ_size(seg_mask, 'X')
    sizeX *= ConstPixelSpacing[1]
    print(f'Время выполнения X: {calcTime} sec')
    print(f'Max размер печени по оси X = {np.round(sizeX, 3)} мм.')

    # Размер органа по оси Y
    sizeY, calcTime = organ_size(seg_mask, 'Y') 
    sizeY *= ConstPixelSpacing[2]
    print(f'Время выполнения Y: {calcTime} sec')
    print(f'Max размер печени по оси Y = {np.round(sizeY, 3)} мм.')

    # Размер органа по оси Z
    startTime = time.time()
    size_z = 0
    for i in range(len(seg_mask)):
        size_z += 1 if np.sum(seg_mask[i] > 0) else 0
    size_z = size_z * ConstPixelSpacing[0]
    print(f'Время выполнения Z: {np.round(time.time() - startTime, 3)} sec')
    print(f'Размер печени по оси Z = {size_z} мм.')

    # Объем органа
    startTime = time.time()
    area = np.sum(seg_mask)
    volume = area * ConstPixelSpacing[1] * ConstPixelSpacing[2] * ConstPixelSpacing[0]
    print(f'Время выполнения расчета объема: {np.round(time.time() - startTime, 3)} sec')
    print(f'Объем печени = {np.round(volume, 3)} мм³. ')

    # Верхняя точка
    # def upper_parall_point(mask: np.array, image_param: tuple):
    #     slice_thickness, pixel_spacing_x, pixel_spacing_y = image_param
    #     for i in range(len(mask)):
    #         for j in range(len(mask[i])):
    #             point_on_parall = np.flatnonzero(mask[i][j])
    #             if len(point_on_parall) > 0:
    #                 x = point_on_parall[len(point_on_parall) // 2] * pixel_spacing_x
    #                 y = j * pixel_spacing_y
    #                 z = i * slice_thickness
    #                 return [x, y, z]
                
    # Нижняя точка
    # def lower_parall_point(mask: np.array, image_param: tuple):
    #     slice_thickness, pixel_spacing_x, pixel_spacing_y = image_param
    #     len_mask = len(mask)
    #     for i in range(len(mask)):
    #         for j in range(len(mask[len_mask - 1 - i])):
    #             point_on_parall = np.flatnonzero(mask[len_mask - 1 - i][j])
    #             if len(point_on_parall) > 0:
    #                 x = point_on_parall[len(point_on_parall) // 2] * pixel_spacing_x
    #                 y = j * pixel_spacing_y
    #                 z = (len_mask - 1 - i) * slice_thickness
    #                 return [x, y, z]
                
    # Дальняя точка
    # def far_parall_point(mask: np.array, image_param: tuple):
    #     slice_thickness, pixel_spacing_x, pixel_spacing_y = image_param
    #     transformed_mask = np.moveaxis(mask, 0, 1)
    #     for i in range(len(transformed_mask)):
    #         for j in range(len(transformed_mask[i])):
    #             point_on_parall = np.flatnonzero(transformed_mask[i][j])
    #             if len(point_on_parall) > 0:
    #                 x = point_on_parall[len(point_on_parall) // 2] * pixel_spacing_x
    #                 y = i * pixel_spacing_y
    #                 z = j * slice_thickness
    #                 return [x, y, z]

    
    # def calc_major_minor():
    #     x1, y1, z1 = upper_parall_point(seg_mask, ConstPixelSpacing) # Верхняя точка
    #     x2, y2, z2 = lower_parall_point(seg_mask, ConstPixelSpacing) # Нижняя точка
    #     x3, y3, z3 = far_parall_point(seg_mask, ConstPixelSpacing) # Дальняя точка
    #     print(f'first point: ({x1}, {y1}, {z1})')
    #     print(f'second point: ({x2}, {y2}, {z2})')
    #     print(f'third point: ({x3}, {y3}, {z3})')
    #     coeffs = [[x1 * x1, y1 * y1, z1 * z1],
    #               [x2 * x2, y2 * y2, z2 * z2],
    #               [x3 * x3, y3 * y3, z3 * z3]]
    #     solutions = [1, 1, 1]
    #     answers = np.linalg.solve(coeffs, solutions)
    #     answers = np.abs(answers)
    #     answers = 1 / answers
    #     answers = np.sqrt(answers)
    #     major = np.max(answers)
    #     minor = np.min(answers)
    #     print(f'major = {major} мм., minor = {minor} мм.')
    #     return answers
    
    def calcMajorMinor(mask: np.array, imageParam: tuple):
        labelledVoxelCoordinates = np.where(mask != 0)
        Np = len(labelledVoxelCoordinates[0])
        coordinates = np.array(labelledVoxelCoordinates, dtype='int').transpose((1, 0))  # Transpose equals zip(*a)
        physicalCoordinates = coordinates * imageParam[1]
        physicalCoordinates -= np.mean(physicalCoordinates, axis = 0)  # Centered at 0
        physicalCoordinates /= np.sqrt(Np)
        covariance = np.dot(physicalCoordinates.T.copy(), physicalCoordinates)
        eigenValues = np.linalg.eigvals(covariance)

        eigenValues.sort()  # Sort the eigenValues from small to large
        if eigenValues[2] < 0 or eigenValues[1] < 0:
            return np.nan
        return np.sqrt(eigenValues[2]) * 4, np.sqrt(eigenValues[1]) * 4, np.sqrt(eigenValues[0]) * 4 # major, minor, middle

    major, minor, middle = calcMajorMinor(seg_mask, ConstPixelSpacing)
    print(f'Мажорный размер = {major} мм.\nМинорный размер = {minor} мм.\n{middle}')
    # Снимок с маской
    pyplot.imshow(dicomMask[680, :, : ], cmap="Greys")
    pyplot.show()

    # 3d модель
    def rotate(elev, angle):
        ax.view_init(elev=elev, azim=angle)

    stl_meshmesh = mesh.Mesh.from_file(PathSTL)
    figure = pyplot.figure()
    ax = figure.add_subplot(projection='3d')
    ax.add_collection3d(mpl_toolkits.mplot3d.art3d.Poly3DCollection(stl_meshmesh.vectors))
    scale = stl_meshmesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('liver')
    # 
    # rx, ry, rz = major, minor, middle
    # Set of all spherical angles:
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    # x = rx / 2 * np.outer(np.cos(u), np.sin(v))
    # y = ry / 2 * np.outer(np.sin(u), np.sin(v))
    # z = rz / 2 * np.outer(np.ones_like(u), np.cos(v))

    # Plot:
    # ax.plot_surface(x, y, z,  rstride = 4, cstride = 4, color = 'g', alpha = 0.2)
    # Adjustment of the axes, so that they all have the same span:
    # max_radius = max(rx, ry, rz)
    # for axis in 'xyz':
    #     getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
    # rotate(90, 0)
    pyplot.show()
    
if __name__ == '__main__':
    imageAnalysis()
    