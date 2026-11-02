#!/usr/bin/env python3
"""Generate the broken concrete slabs used for the collapse rubble.

The repo had no concrete asset, and a box does not read as a broken slab, so the
geometry is generated here: each slab's perimeter is pushed in and out by
band-limited noise, one corner is chipped away, and the top and bottom faces are
slightly warped (slabs rest on uneven rubble, not on a floor). Normals are flat,
which is what spalled concrete looks like.

Three sizes are generated, each into its own model directory, so the rubble is
not one shape repeated:

    concrete_slab         1.90 x 1.25 x 0.18 m   4 rods of rebar
    concrete_slab_wide    2.70 x 1.45 x 0.22 m   5 rods
    concrete_slab_chunk   1.10 x 0.82 x 0.26 m   2 rods

Albedo and normal maps are generated too - mottled grey with stains, aggregate
speckles and hairline cracks - so nothing has to be downloaded and the models
load offline like the rest of this repository.

Deterministic: fixed seeds, sums of sines. Regenerating gives identical files.

    python3 models/concrete_slab/generate_slab_mesh.py

Writes meshes/*.dae and materials/textures/concrete_{albedo,normal}.png into
models/concrete_slab, models/concrete_slab_wide and models/concrete_slab_chunk.
"""

import math
import os
import random
import struct
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.dirname(HERE)

# Slab parameters. main() sets these per variant, so the helpers below always
# work on the slab currently being generated.
L, WD, TH = 1.90, 1.25, 0.18     # length along x, width along y, thickness
N = 28                           # perimeter points
SEED = 4711
UV_TILE = 1.4                    # metres per texture repeat
BAND = 0.055                     # how far the broken edges step in and out
CHIP = (1, 1, 0.52, 0.55)        # chipped corner: x sign, y sign, radius, depth


def noise(s, phase):
    return (0.60 * math.sin(3.1 * s + phase)
            + 0.28 * math.sin(7.3 * s + 2.1 * phase)
            + 0.14 * math.sin(13.7 * s + 3.0 * phase))


def outline():
    """Perimeter of the slab: a rectangle with broken, uneven edges and the
    (+x, +y) corner chipped off."""
    pts = []
    per = 2 * (L + WD)
    for i in range(N):
        d = per * i / N
        if d < L:
            x, y, nx, ny = -L / 2 + d, -WD / 2, 0.0, -1.0
        elif d < L + WD:
            x, y, nx, ny = L / 2, -WD / 2 + (d - L), 1.0, 0.0
        elif d < 2 * L + WD:
            x, y, nx, ny = L / 2 - (d - L - WD), WD / 2, 0.0, 1.0
        else:
            x, y, nx, ny = -L / 2, WD / 2 - (d - 2 * L - WD), -1.0, 0.0

        off = BAND * noise(d * 1.7, 1.3)
        x += nx * off
        y += ny * off

        # chip a corner: pull nearby points towards the corner centre, so the
        # slab breaks off in one bite with a ragged edge left around it
        sx, sy, chip_r, chip_k = CHIP
        cx, cy = sx * (L / 2 - 0.02), sy * (WD / 2 - 0.02)
        r = math.hypot(x - cx, y - cy)
        if r < chip_r:
            k = (chip_r - r) / chip_r
            x += (cx - x) * chip_k * k
            y += (cy - y) * chip_k * k
        pts.append((x, y, d))
    return pts


def top_z(x, y):
    return (TH / 2
            + 0.011 * math.sin(1.7 * x + 0.6) * math.cos(1.9 * y + 1.1)
            + 0.006 * math.sin(4.3 * x - 2.7 * y))


def bottom_z(x, y):
    return (-TH / 2
            + 0.008 * math.cos(1.3 * x - 0.9) * math.sin(2.1 * y + 0.4))


def build_triangles():
    """(p0, p1, p2, uv0, uv1, uv2) per triangle, wound outward."""
    per = outline()
    k = len(per)
    cx = sum(p[0] for p in per) / k
    cy = sum(p[1] for p in per) / k

    top = [(x, y, top_z(x, y)) for x, y, _ in per]
    bot = [(x, y, bottom_z(x, y)) for x, y, _ in per]
    ctop = (cx, cy, top_z(cx, cy) + 0.006)
    cbot = (cx, cy, bottom_z(cx, cy) - 0.004)

    def uv_plane(p):
        return ((p[0] + L / 2) / UV_TILE, (p[1] + WD / 2) / UV_TILE)

    def uv_side(p, d):
        return (d / UV_TILE, (p[2] + TH / 2) / UV_TILE)

    out = []

    def emit(a, b, c, ref, uva, uvb, uvc):
        u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        n = (u[1] * v[2] - u[2] * v[1],
             u[2] * v[0] - u[0] * v[2],
             u[0] * v[1] - u[1] * v[0])
        ln = math.sqrt(sum(x * x for x in n)) or 1.0
        n = (n[0] / ln, n[1] / ln, n[2] / ln)
        if n[0] * ref[0] + n[1] * ref[1] + n[2] * ref[2] < 0.0:
            b, c = c, b
            uvb, uvc = uvc, uvb
            n = (-n[0], -n[1], -n[2])
        out.append((a, b, c, uva, uvb, uvc, n))

    for i in range(k):
        j = (i + 1) % k
        emit(ctop, top[i], top[j], (0.0, 0.0, 1.0),
             uv_plane(ctop), uv_plane(top[i]), uv_plane(top[j]))
        emit(cbot, bot[j], bot[i], (0.0, 0.0, -1.0),
             uv_plane(cbot), uv_plane(bot[j]), uv_plane(bot[i]))

        d_i, d_j = per[i][2], per[j][2]
        if d_j < d_i:
            d_j = d_i + 0.05
        mx = 0.5 * (top[i][0] + top[j][0]) - cx
        my = 0.5 * (top[i][1] + top[j][1]) - cy
        ref = (mx, my, 0.0)
        emit(bot[i], bot[j], top[j], ref,
             uv_side(bot[i], d_i), uv_side(bot[j], d_j), uv_side(top[j], d_j))
        emit(bot[i], top[j], top[i], ref,
             uv_side(bot[i], d_i), uv_side(top[j], d_j), uv_side(top[i], d_i))
    return out


def write_dae(path, tris):
    positions, normals, uvs = [], [], []
    for a, b, c, uva, uvb, uvc, n in tris:
        for p, uv in ((a, uva), (b, uvb), (c, uvc)):
            positions.append(p)
            normals.append(n)
            uvs.append(uv)
    nv = len(positions)

    def flat(values):
        return " ".join("%.5f" % v for p in values for v in p)

    xml = """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <contributor><authoring_tool>generate_slab_mesh.py</authoring_tool></contributor>
    <unit meter="1" name="meter"/>
    <up_axis>Z_UP</up_axis>
  </asset>
  <library_geometries>
    <geometry id="concrete_slab" name="concrete_slab">
      <mesh>
        <source id="concrete_slab-pos">
          <float_array id="concrete_slab-pos-array" count="{n3}">{pos}</float_array>
          <technique_common>
            <accessor source="#concrete_slab-pos-array" count="{nv}" stride="3">
              <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <source id="concrete_slab-nrm">
          <float_array id="concrete_slab-nrm-array" count="{n3}">{nrm}</float_array>
          <technique_common>
            <accessor source="#concrete_slab-nrm-array" count="{nv}" stride="3">
              <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <source id="concrete_slab-uv">
          <float_array id="concrete_slab-uv-array" count="{n2}">{uv}</float_array>
          <technique_common>
            <accessor source="#concrete_slab-uv-array" count="{nv}" stride="2">
              <param name="S" type="float"/><param name="T" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="concrete_slab-vtx">
          <input semantic="POSITION" source="#concrete_slab-pos"/>
        </vertices>
        <triangles material="concrete" count="{nt}">
          <input semantic="VERTEX" source="#concrete_slab-vtx" offset="0"/>
          <input semantic="NORMAL" source="#concrete_slab-nrm" offset="0"/>
          <input semantic="TEXCOORD" source="#concrete_slab-uv" offset="0" set="0"/>
          <p>{p}</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="concrete_slab" name="concrete_slab">
        <instance_geometry url="#concrete_slab"/>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
""".format(nv=nv, n3=nv * 3, n2=nv * 2, nt=len(tris),
           pos=flat(positions), nrm=flat(normals), uv=flat(uvs),
           p=" ".join(str(i) for i in range(nv)))
    with open(path, "w") as f:
        f.write(xml)
    return nv, len(tris)


def write_png_rgb(path, size, rgb):
    """8-bit RGB PNG from a flat row-major bytearray (three bytes per pixel)."""
    raw = bytearray()
    for y in range(size):
        raw.append(0)
        raw += rgb[y * size * 3:(y + 1) * size * 3]

    def chunk(t, d):
        return (struct.pack('>I', len(d)) + t + d
                + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff))

    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(bytes(raw), 9))
           + chunk(b'IEND', b''))
    with open(path, "wb") as f:
        f.write(png)


def write_png(path, size, grey):
    rgb = bytearray()
    for g in grey:
        g = 0.0 if g < 0.0 else (1.0 if g > 1.0 else g)
        rgb.append(min(255, int(g * 1.02 * 255)))
        rgb.append(min(255, int(g * 0.99 * 255)))
        rgb.append(min(255, int(g * 0.94 * 255)))
    write_png_rgb(path, size, rgb)


def write_normal_png(path, grey, size, strength=6.0):
    """Tangent-space normal map derived from the albedo, so the trowelled
    surface and the aggregate catch the light instead of reading as flat grey."""
    rgb = bytearray()
    for y in range(size):
        row = y * size
        up = ((y + 1) % size) * size
        dn = ((y - 1) % size) * size
        for x in range(size):
            dx = grey[row + (x + 1) % size] - grey[row + (x - 1) % size]
            dy = grey[up + x] - grey[dn + x]
            nx, ny, nz = -dx * strength, -dy * strength, 1.0
            ln = math.sqrt(nx * nx + ny * ny + nz * nz)
            rgb.append(int((nx / ln * 0.5 + 0.5) * 255))
            rgb.append(int((ny / ln * 0.5 + 0.5) * 255))
            rgb.append(int((nz / ln * 0.5 + 0.5) * 255))
    write_png_rgb(path, size, rgb)


def concrete_texture(size=512):
    rnd = random.Random(SEED)
    grey = [0.0] * (size * size)
    tau = 2.0 * math.pi
    for y in range(size):
        v = y / size * tau
        row = y * size
        for x in range(size):
            u = x / size * tau
            grey[row + x] = (0.52
                             + 0.045 * math.sin(2 * u + 0.7) * math.cos(3 * v + 1.1)
                             + 0.025 * math.sin(6 * u + 2.3)
                             + 0.020 * math.cos(5 * v - 0.4))

    # damp stains
    for _ in range(12):
        sx, sy = rnd.random() * size, rnd.random() * size
        r = 30.0 + rnd.random() * 110.0
        dark = 0.05 + rnd.random() * 0.10
        for yy in range(int(max(0, sy - r)), int(min(size, sy + r))):
            for xx in range(int(max(0, sx - r)), int(min(size, sx + r))):
                d = math.hypot(xx - sx, yy - sy) / r
                if d < 1.0:
                    grey[yy * size + xx] -= dark * (1.0 - d * d) ** 2

    # aggregate speckles and fine grain, wrapped so the tile stays seamless
    for _ in range(1100):
        x, y = rnd.randrange(size), rnd.randrange(size)
        tone = (rnd.random() - 0.5) * 0.11
        for dx in (0, 1):
            for dy in (0, 1):
                grey[((y + dy) % size) * size + ((x + dx) % size)] += tone
    for _ in range(12000):
        grey[rnd.randrange(size) * size + rnd.randrange(size)] += (rnd.random() - 0.5) * 0.13

    # hairline cracks: random walks across the tile, stamped at all nine wrapped
    # offsets so the pattern stays seamless
    for _ in range(7):
        x, y = rnd.random() * size, rnd.random() * size
        ang = rnd.random() * tau
        for _step in range(170):
            ang += (rnd.random() - 0.5) * 0.55
            x += math.cos(ang) * 2.6
            y += math.sin(ang) * 2.6
            dark = 0.10 + rnd.random() * 0.09
            for ox in (-size, 0, size):
                for oy in (-size, 0, size):
                    xx, yy = int(x) + ox, int(y) + oy
                    if 0 <= xx < size and 0 <= yy < size:
                        grey[yy * size + xx] -= dark
    return grey


VARIANTS = [
    # d: model directory under models/ ("" = this one)
    dict(d="", mesh="concrete_slab", L=1.90, WD=1.25, TH=0.18, seed=4711,
         n=28, band=0.055, chip=(1, 1, 0.52, 0.55), uv=1.4,
         note="1.90 x 1.25 x 0.18, chipped +x/+y corner"),
    dict(d="concrete_slab_wide", mesh="concrete_slab_wide", L=2.70, WD=1.45,
         TH=0.22, seed=8123, n=34, band=0.090, chip=(-1, -1, 0.62, 0.48),
         uv=1.9, note="2.70 x 1.45 x 0.22, broken -x/-y corner"),
    dict(d="concrete_slab_chunk", mesh="concrete_slab_chunk", L=1.10, WD=0.82,
         TH=0.26, seed=3301, n=22, band=0.080, chip=(-1, 1, 0.44, 0.60),
         uv=1.1, note="1.10 x 0.82 x 0.26, heavy break on both ends"),
]


def main():
    global L, WD, TH, SEED, N, UV_TILE, BAND, CHIP
    for v in VARIANTS:
        L, WD, TH = v["L"], v["WD"], v["TH"]
        SEED, N = v["seed"], v["n"]
        UV_TILE, BAND, CHIP = v["uv"], v["band"], v["chip"]

        out = os.path.join(MODELS, v["d"]) if v["d"] else HERE
        tex = os.path.join(out, "materials", "textures")
        os.makedirs(os.path.join(out, "meshes"), exist_ok=True)
        os.makedirs(tex, exist_ok=True)

        nv, nt = write_dae(os.path.join(out, "meshes", v["mesh"] + ".dae"),
                           build_triangles())
        grey = concrete_texture()
        write_png(os.path.join(tex, "concrete_albedo.png"), 512, grey)
        write_normal_png(os.path.join(tex, "concrete_normal.png"), grey, 512)
        print("%-22s %5d verts %5d tris   %s" % (v["mesh"] + ".dae", nv, nt,
                                                 v["note"]))


if __name__ == "__main__":
    main()
