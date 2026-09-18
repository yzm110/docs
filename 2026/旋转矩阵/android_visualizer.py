# 3D visualization for android_euler2dcm  —  tkinter GUI
#
# Android right-hand coordinate system:
#   X = lateral  (positive = right)   -> Pitch axis
#   Y = forward  (positive = forward) -> Roll  axis
#   Z = up       (positive = up)      -> Yaw   axis
#
# Car front always faces Y+ (forward axis)

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict
import sys, os, importlib

# PyInstaller-compatible import for android_euler2dcm
def _load_android_euler2dcm():
    # When frozen by PyInstaller, _MEIPASS is the temp extraction dir
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    if base not in sys.path:
        sys.path.insert(0, base)
    mod = importlib.import_module('android_euler2dcm')
    return mod.android_euler2dcm

android_euler2dcm = _load_android_euler2dcm()


# ── Car model ─────────────────────────────────────────────────────────────────

def _tri(p0,p1,p2):       return [[p0,p1,p2]]
def _quad(p0,p1,p2,p3):   return [[p0,p1,p2,p3]]

def draw_car(ax, alpha=0.20):
    polys, colors, alphas = [], [], []

    def add(faces, color, a=alpha):
        for f in faces:
            polys.append(f); colors.append(color); alphas.append(a)

    W=0.22; by0=-0.50; by1=0.50; bz0=-0.13; bz1=0.10
    trunk_base=-0.20; roof_rear=-0.08; roof_front=0.22; wscreen=0.38; rz=0.26

    for sx in [-W, W]:
        add(_quad([sx,by0,bz0],[sx,by1,bz0],[sx,by1,bz1],[sx,by0,bz1]), 'deepskyblue')
        add(_tri ([sx,by0,bz1],[sx,trunk_base,bz1],[sx,roof_rear,rz]),   'deepskyblue')
        add(_quad([sx,roof_rear,rz],[sx,roof_front,rz],[sx,wscreen,bz1],[sx,trunk_base,bz1]), 'deepskyblue')
        add(_tri ([sx,wscreen,bz1],[sx,roof_front,rz],[sx,by1,bz1]),     'deepskyblue')

    add(_quad([-W,by0,bz0],[W,by0,bz0],[W,by1,bz0],[-W,by1,bz0]),       'deepskyblue')
    add(_quad([-W,by1,bz0],[W,by1,bz0],[W,by1,bz1],[-W,by1,bz1]),       'gold',      a=0.60)
    add(_quad([-W,by0,bz0],[W,by0,bz0],[W,by0,bz1],[-W,by0,bz1]),       'tomato',    a=0.55)
    add(_quad([-W,wscreen,bz1],[W,wscreen,bz1],[W,by1,bz1],[-W,by1,bz1]),'deepskyblue')
    add(_quad([-W,wscreen,bz1],[W,wscreen,bz1],[W,roof_front,rz],[-W,roof_front,rz]), 'lightcyan', a=0.50)
    add(_quad([-W,roof_rear,rz],[W,roof_rear,rz],[W,roof_front,rz],[-W,roof_front,rz]),'deepskyblue')
    add(_quad([-W,trunk_base,bz1],[W,trunk_base,bz1],[W,roof_rear,rz],[-W,roof_rear,rz]),'lightcyan',a=0.45)
    add(_quad([-W,by0,bz1],[W,by0,bz1],[W,trunk_base,bz1],[-W,trunk_base,bz1]),'deepskyblue')

    for hx in [-0.14,0.14]:
        add(_quad([hx-0.055,by1+0.001,bz0+0.08],[hx+0.055,by1+0.001,bz0+0.08],
                  [hx+0.055,by1+0.001,bz0+0.17],[hx-0.055,by1+0.001,bz0+0.17]),'yellow',a=0.90)
    for tx in [-0.14,0.14]:
        add(_quad([tx-0.055,by0-0.001,bz0+0.08],[tx+0.055,by0-0.001,bz0+0.08],
                  [tx+0.055,by0-0.001,bz0+0.17],[tx-0.055,by0-0.001,bz0+0.17]),'red',a=0.90)
    add(_quad([-0.18,by1+0.001,bz0+0.01],[0.18,by1+0.001,bz0+0.01],
              [0.18,by1+0.001,bz0+0.07],[-0.18,by1+0.001,bz0+0.07]),'dimgray',a=0.70)

    theta=np.linspace(0,2*np.pi,32); r=0.10
    for wx in [-W,W]:
        for wy in [-0.32,0.32]:
            xs=np.full_like(theta,wx); ys=wy+r*np.cos(theta); zs=bz0+r*np.sin(theta)
            polys.append(list(zip(xs,ys,zs))); colors.append('dimgray'); alphas.append(alpha+0.30)

    groups = defaultdict(list)
    for face,c,a in zip(polys,colors,alphas):
        groups[(c,round(a,3))].append(face)
    for (c,a),faces in groups.items():
        ax.add_collection3d(Poly3DCollection(faces,alpha=a,
                            facecolor=c,edgecolor='steelblue',linewidth=0.25))


# ── Coordinate frame ──────────────────────────────────────────────────────────

def _draw_frame(ax, R, origin, length=0.85, alpha=1.0, linestyle='-', labels=None):
    colors=['r','g','b']
    if labels is None: labels=['X','Y','Z']
    for i in range(3):
        axis=R[:,i]*length
        ax.quiver(*origin,*axis,color=colors[i],alpha=alpha,
                  linestyle=linestyle,arrow_length_ratio=0.18)
        ax.text(*(np.array(origin)+axis*1.22),labels[i],
                color=colors[i],fontsize=9,fontweight='bold',alpha=alpha)

def _setup_ax(ax, title):
    ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.1,1.1); ax.set_zlim(-1.1,1.1)
    ax.set_xlabel("X (right/pitch)",fontsize=7)
    ax.set_ylabel("Y (forward/roll)",fontsize=7)
    ax.set_zlabel("Z (up/yaw)",fontsize=7)
    ax.set_box_aspect([1,1,1])
    ax.set_title(title,fontsize=8)
    ax.view_init(elev=22,azim=30)
    ax.set_facecolor('#f0f0f0')
    ax.tick_params(colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.zaxis.label.set_color('black')
    ax.title.set_color('black')

def _render(ax1, ax2, roll, pitch, yaw):
    dcm = android_euler2dcm([roll, pitch, yaw])
    origin = [0.,0.,0.]
    for ax in [ax1, ax2]:
        ax.cla()
        draw_car(ax)
    _draw_frame(ax1,np.eye(3),origin,alpha=0.2,linestyle='--',labels=['Xa','Ya','Za'])
    _draw_frame(ax1,dcm,origin,labels=['Xs','Ys','Zs'])
    _setup_ax(ax1,"Before rotation  —  sensor/body frame\n"
                  "Solid=sensor   Faded=Android ref  |  Yellow=FRONT  Red=REAR")
    _draw_frame(ax2,dcm,origin,alpha=0.2,linestyle='--',labels=['Xs','Ys','Zs'])
    _draw_frame(ax2,np.eye(3),origin,labels=['Xa','Ya','Za'])
    _setup_ax(ax2,"After rotation  —  Android standard frame\n"
                  "Solid=Android (X=right,Y=fwd,Z=up)  |  Yellow=FRONT  Red=REAR")
    return dcm

def _fill_matrix_table(table_frame, dcm):
    """Fill a pre-built 4x4 grid of Labels with DCM values."""
    for widget in table_frame.winfo_children():
        widget.destroy()

    headers = ['row/col', 'C1', 'C2', 'C3']
    row_labels = ['R1', 'R2', 'R3']

    # Style
    HDR_BG  = '#2d2d50';  HDR_FG  = '#aaaaff'
    CEL_BG  = '#1e1e2e';  CEL_FG  = '#00ff99'
    HDR_FONT = ('Arial', 9, 'bold')
    CEL_FONT = ('Courier', 9)

    # Header row
    for c, h in enumerate(headers):
        tk.Label(table_frame, text=h,
                 bg=HDR_BG, fg=HDR_FG, font=HDR_FONT,
                 width=11, relief='flat', pady=5
                 ).grid(row=0, column=c, padx=1, pady=1, sticky='nsew')

    # Data rows
    for r in range(3):
        # Row label
        tk.Label(table_frame, text=row_labels[r],
                 bg=HDR_BG, fg=HDR_FG, font=HDR_FONT,
                 width=11, relief='flat', pady=5
                 ).grid(row=r+1, column=0, padx=1, pady=1, sticky='nsew')
        # Values
        for c in range(3):
            tk.Label(table_frame, text=f"{dcm[r,c]:+.6f}",
                     bg=CEL_BG, fg=CEL_FG, font=CEL_FONT,
                     width=12, relief='flat', pady=5
                     ).grid(row=r+1, column=c+1, padx=1, pady=1, sticky='nsew')


# ── Main GUI ──────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.title("Android Euler → DCM Visualizer")
    root.configure(bg='#1e1e2e')
    root.geometry("1600x750")

    # ── Left: matplotlib figure ───────────────────────────────────
    fig = plt.Figure(figsize=(10, 6), facecolor='#e8e8e8')
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    fig.suptitle("R = Rz(yaw) · Ry(roll) · Rx(pitch)",
                 color='black', fontsize=10)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ── Right: control panel ──────────────────────────────────────
    panel = tk.Frame(root, bg='#1e1e2e', width=420)
    panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
    panel.pack_propagate(False)

    tk.Label(panel, text="Android Euler → DCM",
             bg='#1e1e2e', fg='white', font=('Arial',12,'bold')).pack(pady=(10,20))

    # Entry rows
    entries = {}
    sliders = {}
    params = [
        ('Roll  (Y fwd)',  'roll',  -180, 180, 19.3),
        ('Pitch (X right)','pitch',  -90,  90,  0.0),
        ('Yaw   (Z up)',   'yaw',  -180, 180, 90.0),
    ]

    updating = [False]   # mutable flag

    for label, key, lo, hi, init in params:
        frm = tk.Frame(panel, bg='#1e1e2e')
        frm.pack(fill=tk.X, pady=6)

        tk.Label(frm, text=label, bg='#1e1e2e', fg='#aaaacc',
                 font=('Arial',10), width=16, anchor='w').pack(side=tk.LEFT)

        var = tk.DoubleVar(value=init)
        entries[key] = var

        ent = tk.Entry(frm, textvariable=var, width=8,
                       bg='white', fg='black', font=('Arial',10),
                       justify='center', relief='flat')
        ent.pack(side=tk.LEFT, padx=(4,0))

        frm2 = tk.Frame(panel, bg='#1e1e2e')
        frm2.pack(fill=tk.X, padx=4, pady=(0,4))

        sl = tk.Scale(frm2, from_=lo, to=hi, orient=tk.HORIZONTAL,
                      variable=var, resolution=0.1,
                      bg='#2a2a4a', fg='white', troughcolor='#555577',
                      highlightthickness=0, length=360, showvalue=False)
        sl.pack(fill=tk.X)
        sliders[key] = sl

    # ── Matrix display (table) ────────────────────────────────────
    tk.Label(panel, text="DCM  (3×3)   v_android = DCM · v_sensor",
             bg='#1e1e2e', fg='#aaaaff',
             font=('Arial', 9, 'bold')).pack(pady=(18, 4))

    table_frame = tk.Frame(panel, bg='#0a0a1a')
    table_frame.pack(padx=6)

    # Integer output: int = (value + 1) * 1000000
    tk.Label(panel, text="Integer output  ( int = (val+1) × 1000000 )",
             bg='#1e1e2e', fg='#aaaaff',
             font=('Arial', 9, 'bold')).pack(pady=(14, 2))

    # 9 read-only Entry widgets in one column, each selectable/copyable
    int_frame = tk.Frame(panel, bg='#1e1e2e')
    int_frame.pack(fill=tk.X, padx=4, pady=(0, 4))

    # Column header row
    hdr = tk.Frame(int_frame, bg='#1e1e2e')
    hdr.pack(fill=tk.X, pady=(0, 2))
    tk.Label(hdr, text="",
             bg='#1e1e2e', font=('Courier', 9, 'bold'),
             width=6, anchor='e').pack(side=tk.LEFT)
    tk.Label(hdr, text="Dec",
             bg='#1e1e2e', fg='#ffcc44',
             font=('Courier', 9, 'bold'), width=14, anchor='w'
             ).pack(side=tk.LEFT, padx=(4, 0))
    tk.Label(hdr, text="Hex",
             bg='#1e1e2e', fg='#44ddcc',
             font=('Courier', 9, 'bold'), width=10, anchor='w'
             ).pack(side=tk.LEFT, padx=(6, 0))

    int_entries = {}   # key: (ri, ci)
    hex_entries = {}   # key: (ri, ci)
    for idx in range(9):
        ri, ci = divmod(idx, 3)
        lbl = f"R{ri+1}C{ci+1}:"
        row_frame = tk.Frame(int_frame, bg='#1e1e2e')
        row_frame.pack(fill=tk.X, pady=1)
        tk.Label(row_frame, text=lbl,
                 bg='#1e1e2e', fg='#aaaaff',
                 font=('Courier', 9), width=6, anchor='e').pack(side=tk.LEFT)
        var = tk.StringVar(value="")
        ent = tk.Entry(row_frame, textvariable=var,
                       width=14, font=('Courier', 9),
                       bg='#2a2a1a', fg='#ffcc44',
                       justify='left', relief='flat')
        ent.pack(side=tk.LEFT, padx=(4, 0))
        ent.bind("<Key>", lambda e: "break")
        int_entries[(ri, ci)] = var
        hvar = tk.StringVar(value="")
        hent = tk.Entry(row_frame, textvariable=hvar,
                        width=10, font=('Courier', 9),
                        bg='#1a2a2a', fg='#44ddcc',
                        justify='left', relief='flat')
        hent.pack(side=tk.LEFT, padx=(6, 0))
        hent.bind("<Key>", lambda e: "break")
        hex_entries[(ri, ci)] = hvar

    # ── Buttons ───────────────────────────────────────────────────
    btn_frame = tk.Frame(panel, bg='#1e1e2e')
    btn_frame.pack(pady=16)

    def on_apply():
        try:
            r = float(entries['roll'].get())
            p = float(entries['pitch'].get())
            y = float(entries['yaw'].get())
        except (tk.TclError, ValueError):
            return
        dcm = _render(ax1, ax2, r, p, y)
        _fill_matrix_table(table_frame, dcm)

        # Integer output: int = (val + 1) * 1000000  →  decimal + hex
        for ri in range(3):
            for ci in range(3):
                val = int(round((dcm[ri, ci] + 1) * 1000000))
                int_entries[(ri, ci)].set(str(val))
                hex_entries[(ri, ci)].set(f"{val & 0xFFFFFF:06X}")

        canvas.draw()

    def on_reset():
        entries['roll'].set(0.0)
        entries['pitch'].set(0.0)
        entries['yaw'].set(0.0)
        on_apply()

    tk.Button(btn_frame, text="  Apply  ",
              bg='#2a6a2a', fg='white', font=('Arial',11,'bold'),
              relief='flat', padx=10, pady=4,
              command=on_apply).pack(side=tk.LEFT, padx=6)

    tk.Button(btn_frame, text="  Reset  ",
              bg='#3a3a6a', fg='white', font=('Arial',11,'bold'),
              relief='flat', padx=10, pady=4,
              command=on_reset).pack(side=tk.LEFT, padx=6)

    # Initial render
    on_apply()
    root.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
