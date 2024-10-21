#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.animation as manimation
import matplotlib.pyplot as plt

import sys

header_t = np.dtype([('Lx', 'i4'), ('Ly', 'i4'), ('nx', 'i4'), ('ny', 'i4'), ('H', 'f8'), ('g', 'f8'), ('r', 'f8'), ('h', 'f8'), ('dt', 'f8'), ('num_iter', 'i4'), ('save_iter', 'i4')])

def main(file1 = 'serial.out', file2 = 'gpu.out', out='correctness.gif'):
    header1 = np.fromfile(file1, dtype=header_t, count=1)
    header2 = np.fromfile(file2, dtype=header_t, count=1)

    if (np.array_equal(header1, header2) == False):
        print("Headers are not equal, so why compare these two runs? They can never be the same.")
        return

    nx = header1['nx'][0]
    ny = header1['ny'][0]

    Lx = header1['Lx'][0]
    Ly = header1['Ly'][0]

    dx = Lx / nx
    dy = Ly / ny

    num_iter = header1['num_iter'][0]
    save_iter = header1['save_iter'][0]

    num_frames = max(num_iter // save_iter, 1)

    h1 = np.fromfile(file1, offset=header_t.itemsize, dtype='f8').reshape((num_frames, nx + 1, ny + 1))
    h2 = np.fromfile(file2, offset=header_t.itemsize, dtype='f8').reshape((num_frames, nx + 1, ny + 1))

    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    x = np.arange(num_frames) * save_iter
    y = np.max(np.abs(h1[:, :-1, :-1] - h2[:, :-1, :-1]), axis=(1, 2))

    print(f"Max error: {np.max(y)}")

    ax1.plot(x, y)

    hx = (-Lx/2 + dx/2.0 + np.arange(nx)*dx)[:, np.newaxis]
    hy = (-Ly/2 + dy/2.0 + np.arange(ny)*dy)[np.newaxis, :]

    X, Y = np.meshgrid(hx, hy)

    def update_anim(i):
        Z = np.abs(h1[i, :-1, :-1].T - h2[i, :-1, :-1].T)

        ax2.cla()
        ax2.contourf(X/Lx, Y/Ly, Z, cmap=plt.cm.RdBu)
        ax2.set_title(f"Error for t = {i*save_iter}")

    animation_fig = manimation.FuncAnimation(fig, update_anim, frames=num_frames, interval=10)
    animation_fig.save(out)

if __name__ == "__main__":
    main(*sys.argv[1:])