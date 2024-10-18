#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

header_t = np.dtype([('Lx', 'i4'), ('Ly', 'i4'), ('nx', 'i4'), ('ny', 'i4'), ('H', 'f8'), ('g', 'f8'), ('r', 'f8'), ('h', 'f8'), ('dt', 'f8'), ('num_iter', 'i4'), ('save_iter', 'i4')])

header = np.fromfile('c.out', dtype=header_t, count=1)

Lx = header['Lx'][0]
Ly = header['Ly'][0]

nx = header['nx'][0]
ny = header['ny'][0]

num_i = header['num_iter'][0]
save_i = header['save_iter'][0]

num_frames = num_i // save_i

print(nx, ny)

h = np.fromfile('c.out', offset=header_t.itemsize, dtype='f8').reshape((num_frames, nx + 1, ny + 1))
# hswe = np.fromfile('swe.out.npy', dtype='f8').reshape((1, nx + 2, ny + 2))
# dh = np.fromfile('dh.out', dtype='f8').reshape((num_frames, nx, ny))
# du = np.fromfile('du.out', dtype='f8').reshape((num_frames, nx + 1, ny))
# dv = np.fromfile('dv.out', dtype='f8').reshape((num_frames, nx, ny + 1))

dx = Lx / nx
dy = Ly / ny

hx = (-Lx/2 + dx/2.0 + np.arange(nx)*dx)[:, np.newaxis]
hy = (-Ly/2 + dy/2.0 + np.arange(ny)*dy)[np.newaxis, :]

plt.ion()
fig = plt.figure(figsize=(10, 10))

nc = 12
colorlevels = np.concatenate([np.linspace(-1, -.05, nc), np.linspace(.05, 1, nc)])

hmax = np.max(np.abs(h[0, :, :]))

for i in range(num_frames):
    plt.clf()

    X, Y = np.meshgrid(hx, hy)

    ax = fig.add_subplot(221)
    ax.contourf(X/Lx, Y/Ly, h[i, :-1, :-1].T, cmap=plt.cm.RdBu, levels=colorlevels*hmax)
    plt.title('h')

    ax = fig.add_subplot(222, projection='3d')
    ax.contourf(X/Lx, Y/Ly, h[i, :-1, :-1].T, cmap=plt.cm.RdBu, levels=colorlevels*hmax)
    plt.title('h')

    # print(np.sum(h[i, :-1, :-1]))

    plt.subplot(223)
    plt.plot(hx/Lx, h[i, :-1, :-1][:, ny//2])
    plt.xlim(-0.5, 0.5)
    plt.ylim(-hmax, hmax)
    plt.title('h along y=0')

    plt.pause(0.0001)
    plt.draw()


# frames_t = np.dtype(('f8', (nx + 2, ny + 2)))

# framesc = np.fromfile('c.out', offset=header_t.itemsize, dtype=frames_t)
# framesswe = np.fromfile('swe.out.npy', dtype=frames_t)

# plt.imshow(framesswe[0][1:-1, 1:-1], cmap='viridis')
# plt.show()

# plt.imshow(framesc[0][1:-1, 1:-1], cmap='viridis')
# plt.show()

# cdh = np.fromfile('c.out', dtype='f8', offset=header_t.itemsize).reshape((nx + 2, ny + 2))
# swedh = np.fromfile('swe.out.npy', dtype='f8').reshape((nx + 2, ny + 2))

# cdh = np.fromfile('du.out', dtype='f8').reshape((nx + 1, ny))
# swedh = np.fromfile('swedu.out.npy', dtype='f8').reshape((nx + 1, ny))

# print(np.max(cdh))

# plt.imshow(cdh)
# plt.show()

# plt.imshow(swedh)
# plt.show()

# plt.imshow(np.abs(cdh - swedh))
# plt.show()

# print(np.max(np.abs(cdh - swedh)))