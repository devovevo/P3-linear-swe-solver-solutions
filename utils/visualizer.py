#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.animation as manimation
import matplotlib.pyplot as plt

import sys

header_t = np.dtype([('Lx', 'i4'), ('Ly', 'i4'), ('nx', 'i4'), ('ny', 'i4'), ('H', 'f8'), ('g', 'f8'), ('r', 'f8'), ('h', 'f8'), ('dt', 'f8'), ('num_iter', 'i4'), ('save_iter', 'i4')])

def main(file='serial.out', animation_file='anim.gif'):
    header = np.fromfile(file, dtype=header_t, count=1)

    Lx = header['Lx'][0]
    Ly = header['Ly'][0]

    nx = header['nx'][0]
    ny = header['ny'][0]

    stride = nx // 2

    num_i = header['num_iter'][0]
    save_i = header['save_iter'][0]

    num_frames = max(num_i // save_i, 1)

    dx = Lx / nx
    dy = Ly / ny

    h = np.fromfile(file, offset=header_t.itemsize, dtype='f8').reshape((num_frames, nx + 1, ny + 1))
    hmax = np.max(np.abs(h[0, :-1, :-1]))

    hx = (-Lx/2 + dx/2.0 + np.arange(nx)*dx)[:, np.newaxis]
    hy = (-Ly/2 + dy/2.0 + np.arange(ny)*dy)[np.newaxis, :]

    X, Y = np.meshgrid(hx, hy)

    # Create the figure and axes objects
    fig = plt.figure(figsize=(15, 15))

    nc = 12
    colorlevels = np.concatenate([np.linspace(-1, -.05, nc), np.linspace(.05, 1, nc)])

    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)

    def update_anim(i):
        Z = h[i, :-1, :-1].T
        
        ax1.cla()
        ax1.contourf(X/Lx, Y/Ly, Z, cmap=plt.cm.RdBu, levels=colorlevels*hmax)
        ax1.set_title(f"h(x, y) for t = {i*save_i}")

        ax2.cla()
        ax2.contourf(X/Lx, Y/Ly, Z, cmap=plt.cm.RdBu, levels=colorlevels*hmax)
        ax2.set_title(f"h(x, y) for t = {i*save_i}")

        ax3.cla()
        ax3.plot(hx/Lx, h[i, :-1, :-1][:, ny//2])
        ax3.set_title(f"h(x, 0) for t = {i*save_i}")
        ax3.set_ylim(-hmax, hmax)

    animation_fig = manimation.FuncAnimation(fig, update_anim, frames=num_frames, interval=10)
    animation_fig.save(animation_file)

if __name__ == "__main__":
    main(*sys.argv[1:])