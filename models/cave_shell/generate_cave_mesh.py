#!/usr/bin/env python3
"""Generate the cave shell meshes for the `cave_shell` model.

The cave is one continuous swept tube: an open mouth in a rock face, a large
gallery, a 1.5 m squeeze, and the small chamber where the trapped person is.
Above the chamber a vertical daylight shaft is cut into the roof, so the chamber
is lit by real light coming down through a hole rather than by an invented lamp.
The mouth is wider than the rest of the passage, and wider on some sides than
others, so it reads as a natural entrance rather than a bored round hole.
Stalactites hang from the roof and stalagmites stand on the floor through the
mouth zone and the gallery (see SPELEOTHEMS below).

Run from the repository root:

    python3 models/cave_shell/generate_cave_mesh.py

Writes, next to this file:

    meshes/cave_shell.dae      the whole shell, used for both visual and
                              collision (one mesh, so nothing can leak through
                              a wall the renderer shows but physics misses)

and prints a floor-height table for placing props (the world file uses it).

Everything is deterministic: no random seeds, no external data. The geometry is
sums of sines, so regenerating gives byte-identical meshes.
"""

import math
import os
import xml.sax.saxutils as sax

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Layout (world ENU metres: x east, y north, z up)
# ---------------------------------------------------------------------------
# Tunnel centreline: (x, y, z, tube radius). The mouth at x = -31 is the open
# end; the far end at x = +9.2 is capped, so the chamber is a dead end. Radii
# pinch for the squeeze at x = 0.6..1.9, which leaves the 1.5 m entrance to the
# chamber (see `squeeze clearance` in the diagnostics).
TUNNEL = [
    (-31.0, -6.4, 2.30, 3.40),   # mouth, open
    (-27.0, -6.1, 2.10, 3.60),
    (-23.0, -5.4, 1.90, 3.90),
    (-19.0, -4.2, 1.70, 4.20),
    (-15.0, -2.3, 1.50, 4.40),
    (-11.0,  0.2, 1.30, 4.40),
    ( -7.5,  2.4, 1.10, 4.00),
    ( -4.5,  3.9, 1.00, 3.20),
    ( -2.4,  4.6, 0.95, 2.20),
    ( -0.7,  5.0, 0.92, 1.15),   # funnel into the squeeze
    (  0.6,  5.3, 0.90, 0.72),   # squeeze starts
    (  1.9,  5.5, 0.90, 0.72),   # squeeze ends
    (  3.6,  5.8, 0.90, 1.80),   # opens into the chamber
    (  5.6,  6.2, 0.92, 2.35),   # chamber centre, under the daylight shaft
    (  7.6,  6.6, 0.95, 2.20),
    (  9.2,  7.0, 1.00, 1.70),   # dead end (capped)
]

# Vertical daylight shaft over the chamber: (z, inner radius), bottom first.
SHAFT_X, SHAFT_Y = 5.6, 6.2
SHAFT = [
    (2.90, 1.45),   # small lip that hides the ragged roof hole from below
    (3.45, 1.10),
    (6.00, 1.05),
    (9.50, 1.00),
    (12.50, 1.05),
    (14.50, 1.45),  # open top
]

STEP = 0.55        # ring spacing along the tube (m)
RING = 44          # vertices per ring
THICKNESS = 1.60   # rock mass between cave interior and outside skin (m)
SHAFT_THICKNESS = 1.20
ROOF_HOLE_R = 1.25   # triangles whose centroid falls inside this are cut away

TAU = 2.0 * math.pi


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mul(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def norm(a):
    n = math.sqrt(dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-12 else (1.0, 0.0, 0.0)


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def catmull_rom(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return tuple(
        0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * t
               + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
               + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3)
        for k in range(3))


# ---------------------------------------------------------------------------
# Rock surface detail
# ---------------------------------------------------------------------------
def wobble(s, theta):
    """Smooth, 2*pi-periodic in theta, band-limited in s. Range about [-1, 1]."""
    w = (0.55 * math.sin(3 * theta + 0.80 * s + 0.40)
         + 0.28 * math.sin(5 * theta - 0.55 * s + 1.90)
         + 0.14 * math.sin(9 * theta + 1.70 * s + 2.60)
         + 0.20 * math.sin(2 * theta - 0.30 * s + 3.40))
    return w / 1.17


def radius_at(r, s, theta):
    """Tube radius with rock relief. Relief fades out in narrow passages, so the
    squeeze stays open to a known clearance whatever the noise does."""
    amp = 0.16 * min(1.0, r / 2.0)
    return r * (1.0 + amp * wobble(s, theta) * (1.0 + 0.35 * math.sin(0.45 * s + 1.1)))


def hill_bulge(s):
    """Rock mass around the mouth swells into a hill, so the entrance sits in a
    hillside instead of at the cut end of a pipe. Tapers to zero at the mouth
    itself, leaving a normal 1.6 m rock rim around the opening."""
    if s > 14.0:
        return 0.0
    return 9.5 * math.sin(math.pi * s / 14.0) ** 1.5


def floor_depth(r):
    """How far the floor is dropped below the centreline. Narrow passages stay
    round (a squeeze is a hole, not a slot); rooms get a broad flat-ish floor."""
    return 0.55 * r * clamp((r - 1.6) / 1.2, 0.0, 1.0)


def mouth_flare(s, theta):
    """Extra radius near the mouth, unevenly distributed around the ring: the
    entrance opens out to one side more than the other instead of being an even
    round bore. Fades to nothing by 14 m in, so the gallery is unaffected."""
    if s > 14.0:
        return 0.0
    falloff = math.cos(clamp(s / 14.0, 0.0, 1.0) * math.pi / 2.0) ** 1.2
    shape = (0.55 + 0.30 * math.sin(2.0 * theta + 0.8)
             + 0.15 * math.sin(3.0 * theta - 1.9))
    return 2.2 * falloff * max(shape, 0.08)


# ---------------------------------------------------------------------------
# Mesh accumulation
# ---------------------------------------------------------------------------
class Mesh:
    def __init__(self):
        # (a, b, c) with each corner a (position, uv) pair, already wound so the
        # face normal points along the reference direction supplied.
        self.tris = []

    def tri(self, a, b, c, ref):
        pa, pb, pc = a[0], b[0], c[0]
        n = cross(sub(pb, pa), sub(pc, pa))
        if dot(n, ref) < 0.0:
            b, c = c, b
            n = mul(n, -1.0)
        if math.sqrt(dot(n, n)) < 1e-9:
            return False
        self.tris.append((a, b, c))
        return True

    def quad(self, a, b, c, d, ref):
        # Split along a-c; both triangles get the same reference direction.
        kept = self.tri(a, b, c, ref)
        kept = self.tri(a, c, d, ref) or kept
        return kept

    def cut(self, predicate):
        """Drop triangles whose centroid ::predicate(centroid):: is True."""
        kept = []
        for t in self.tris:
            cen = mul(add(add(t[0][0], t[1][0]), t[2][0]), 1.0 / 3.0)
            if not predicate(cen):
                kept.append(t)
        self.tris = kept

    # -- COLLADA ------------------------------------------------------------
    def write_dae(self, path, node_name, material_name):
        # Smooth normals: area-weighted face normals accumulated per position.
        # Vertices are non-indexed (three per triangle), which keeps the writer
        # simple; the files are a couple of MB, which is fine for static rock.
        acc = {}
        key_of = {}

        def vkey(p):
            return (round(p[0], 4), round(p[1], 4), round(p[2], 4))

        for a, b, c in self.tris:
            pa, pb, pc = a[0], b[0], c[0]
            n = cross(sub(pb, pa), sub(pc, pa))
            for corner in (a, b, c):
                k = vkey(corner[0])
                acc[k] = add(acc.get(k, (0.0, 0.0, 0.0)), n)

        positions, normals, uvs, indices = [], [], [], []
        for a, b, c in self.tris:
            for corner in (a, b, c):
                p, uv = corner
                n = norm(acc[vkey(p)])
                indices.append(len(positions))
                positions.append(p)
                normals.append(n)
                uvs.append(uv)

        def floats(points):
            return " ".join(" ".join("%.5f" % v for v in p) for p in points)

        nv = len(positions)
        nt = len(self.tris)
        body = []
        body.append('          <float_array id="%s-pos-array" count="%d">%s</float_array>'
                    % (node_name, nv * 3, floats(positions)))
        body.append('          <float_array id="%s-nrm-array" count="%d">%s</float_array>'
                    % (node_name, nv * 3, floats(normals)))
        body.append('          <float_array id="%s-uv-array" count="%d">%s</float_array>'
                    % (node_name, nv * 2, floats(uvs)))
        pidx = " ".join("%d %d %d" % (i, i, i) for i in indices)

        xml = """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <contributor>
      <authoring_tool>generate_cave_mesh.py</authoring_tool>
    </contributor>
    <unit meter="1" name="meter"/>
    <up_axis>Z_UP</up_axis>
  </asset>
  <library_materials>
    <material id="{mat}-material" name="{mat}">
      <instance_effect url="#{mat}-effect"/>
    </material>
  </library_materials>
  <library_effects>
    <effect id="{mat}-effect">
      <profile_COMMON>
        <technique sid="common">
          <lambert>
            <diffuse>
              <color>0.42 0.40 0.37 1</color>
            </diffuse>
          </lambert>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>
  <library_geometries>
    <geometry id="{node}" name="{node}">
      <mesh>
        <source id="{node}-pos">
          <float_array id="{node}-pos-array" count="0"/>
          <technique_common>
            <accessor source="#{node}-pos-array" count="0" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <source id="{node}-nrm">
          <float_array id="{node}-nrm-array" count="0"/>
          <technique_common>
            <accessor source="#{node}-nrm-array" count="0" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <source id="{node}-uv">
          <float_array id="{node}-uv-array" count="0"/>
          <technique_common>
            <accessor source="#{node}-uv-array" count="0" stride="2">
              <param name="S" type="float"/>
              <param name="T" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="{node}-vtx">
          <input semantic="POSITION" source="#{node}-pos"/>
        </vertices>
        <triangles material="{mat}" count="{nt}">
          <input semantic="VERTEX" source="#{node}-vtx" offset="0"/>
          <input semantic="NORMAL" source="#{node}-nrm" offset="0"/>
          <input semantic="TEXCOORD" source="#{node}-uv" offset="0" set="0"/>
          <p>{pidx}</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="{node}-node" name="{node}" type="NODE">
        <instance_geometry url="#{node}">
          <bind_material>
            <technique_common>
              <instance_material symbol="{mat}" target="#{mat}-material"/>
            </technique_common>
          </bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene>
    <instance_visual_scene url="#Scene"/>
  </scene>
</COLLADA>
""".format(node=node_name, mat=material_name, nt=nt, pidx=pidx)

        # Splice the arrays in and fix the accessor counts.
        xml = xml.replace(
            '<float_array id="%s-pos-array" count="0"/>' % node_name,
            body[0])
        xml = xml.replace(
            '<float_array id="%s-nrm-array" count="0"/>' % node_name,
            body[1])
        xml = xml.replace(
            '<float_array id="%s-uv-array" count="0"/>' % node_name,
            body[2])
        xml = xml.replace(
            '<accessor source="#%s-pos-array" count="0"' % node_name,
            '<accessor source="#%s-pos-array" count="%d"' % (node_name, nv))
        xml = xml.replace(
            '<accessor source="#%s-nrm-array" count="0"' % node_name,
            '<accessor source="#%s-nrm-array" count="%d"' % (node_name, nv))
        xml = xml.replace(
            '<accessor source="#%s-uv-array" count="0"' % node_name,
            '<accessor source="#%s-uv-array" count="%d"' % (node_name, nv))

        with open(path, "w") as f:
            f.write(xml)
        return nv, nt


# ---------------------------------------------------------------------------
# Sweeping the tunnel
# ---------------------------------------------------------------------------
def sample_path(ctrl, step):
    """Centripetal Catmull-Rom through the control points, resampled to a
    uniform arc-length spacing. Radius is piecewise linear and then smoothed,
    which keeps the squeeze the width it was asked for instead of overshooting
    into an accidental obstruction."""
    dense = []
    n = len(ctrl)
    for i in range(n - 1):
        p0 = ctrl[max(i - 1, 0)]
        p1 = ctrl[i]
        p2 = ctrl[i + 1]
        p3 = ctrl[min(i + 2, n - 1)]
        seg = math.dist(p1[:3], p2[:3])
        m = max(4, int(math.ceil(seg / (step / 6.0))))
        for j in range(m):
            t = j / float(m)
            pos = catmull_rom(p0[:3], p1[:3], p2[:3], p3[:3], t)
            r = p1[3] + (p2[3] - p1[3]) * t
            dense.append((pos[0], pos[1], pos[2], r))
    dense.append(ctrl[-1])

    # Arc length, then uniform resampling.
    cum = [0.0]
    for a, b in zip(dense, dense[1:]):
        cum.append(cum[-1] + math.dist(a[:3], b[:3]))
    total = cum[-1]
    count = int(total / step) + 1
    out = []
    j = 0
    for i in range(count):
        target = total * i / float(count - 1)
        while j < len(cum) - 2 and cum[j + 1] < target:
            j += 1
        span = cum[j + 1] - cum[j]
        t = 0.0 if span <= 0 else (target - cum[j]) / span
        a, b = dense[j], dense[j + 1]
        out.append(((a[0] + (b[0] - a[0]) * t,
                     a[1] + (b[1] - a[1]) * t,
                     a[2] + (b[2] - a[2]) * t),
                    a[3] + (b[3] - a[3]) * t))

    # Smooth the radius so the funnel into the squeeze is not a hard step.
    radii = [p[1] for p in out]
    for _ in range(3):
        radii = [
            (radii[max(i - 1, 0)] + 2.0 * radii[i] + radii[min(i + 1, len(radii) - 1)]) / 4.0
            for i in range(len(radii))
        ]
    return [(p[0], r) for p, r in zip(out, radii)], total


def frames(points):
    """World-up framed axes per station. The tunnel stays near horizontal, so a
    fixed up reference gives a floor that never rolls under the drone."""
    out = []
    n = len(points)
    for i, p in enumerate(points):
        ahead = points[min(i + 1, n - 1)]
        behind = points[max(i - 1, 0)]
        t = norm(sub(ahead, behind))
        up_ref = (0.0, 0.0, 1.0)
        if abs(t[2]) > 0.995:            # vertical section (daylight shaft)
            a_axis = (1.0, 0.0, 0.0)
            b_axis = (0.0, 1.0, 0.0)
        else:
            b_axis = norm(sub(up_ref, mul(t, dot(t, up_ref))))
            a_axis = norm(cross(b_axis, t))
        out.append((p, t, a_axis, b_axis))
    return out


def ring_points(center, a_axis, b_axis, r, s, theta_offset=0.0, flare=None):
    pts = []
    for j in range(RING):
        theta = TAU * j / RING + theta_offset
        rr = radius_at(r, s, theta - theta_offset)
        if flare is not None:
            rr += flare(theta - theta_offset)
        al = rr * math.cos(theta - theta_offset)
        bl = rr * math.sin(theta - theta_offset)
        f = floor_depth(r)
        if f > 0.0 and bl < -f:
            bl = -(f + (-bl - f) * 0.15)
        pts.append(add(center, add(mul(a_axis, al), mul(b_axis, bl))))
    return pts


def roof_hole(c):
    """True for triangles that sit in the daylight-shaft opening."""
    return (math.hypot(c[0] - SHAFT_X, c[1] - SHAFT_Y) < ROOF_HOLE_R
            and c[2] > 1.90)


def build_tunnel():
    stations, total = sample_path(TUNNEL, STEP)
    fr = frames([p for p, _ in stations])
    s_values, inner, outer = [], [], []
    s = 0.0
    for i, ((p, r), (c, t, a_axis, b_axis)) in enumerate(zip(stations, fr)):
        if i > 0:
            s += math.dist(p, stations[i - 1][0])
        s_values.append(s)
        inner.append(ring_points(c, a_axis, b_axis, r, s,
                                 flare=lambda th: mouth_flare(s, th)))
        # Thicker skin with its own relief phase, so the rock mass outside does
        # not look like a copy of the cave it wraps, and a hill over the mouth
        # so the entrance reads as a cave in a hillside from outside. The mouth
        # flare is applied to both, or the rock rim would stay circular while
        # the passage opened out behind it.
        outer.append(ring_points(c, a_axis, b_axis,
                                 r + THICKNESS + hill_bulge(s), s + 41.0,
                                 theta_offset=0.13,
                                 flare=lambda th: mouth_flare(s, th)))
    return inner, outer, s_values, total, fr


def tunnel_mesh():
    inner, outer, s_values, total, fr = build_tunnel()
    m = Mesh()
    n_st = len(inner)

    for i in range(n_st - 1):
        c0, t0, _, _ = fr[i]
        c1, t1, _, _ = fr[i + 1]
        for j in range(RING):
            jn = (j + 1) % RING
            u0, u1 = s_values[i] / 3.0, s_values[i + 1] / 3.0
            v0, v1 = j / float(RING) * 4.0, (j + 1) / float(RING) * 4.0
            a = (inner[i][j], (u0, v0))
            b = (inner[i + 1][j], (u1, v0))
            c = (inner[i + 1][jn], (u1, v1))
            d = (inner[i][jn], (u0, v1))
            # Face the cave axis: visible from inside the cave.
            ref = sub(c0, inner[i][j])
            m.quad(a, b, c, d, ref)

    # End wall at the far end, seen from inside.
    last_center = fr[-1][0]
    for j in range(RING):
        jn = (j + 1) % RING
        a = (inner[-1][j], (0.0, j / float(RING) * 4.0))
        b = (inner[-1][jn], (1.0, (j + 1) / float(RING) * 4.0))
        c = (last_center, (0.5, 0.5))
        m.tri(a, b, c, fr[-1][1])

    for i in range(n_st - 1):
        c0 = fr[i][0]
        for j in range(RING):
            jn = (j + 1) % RING
            u0, u1 = s_values[i] / 3.0 + 1.7, s_values[i + 1] / 3.0 + 1.7
            v0, v1 = j / float(RING) * 4.0, (j + 1) / float(RING) * 4.0
            a = (outer[i][j], (u0, v0))
            b = (outer[i + 1][j], (u1, v0))
            c = (outer[i + 1][jn], (u1, v1))
            d = (outer[i][jn], (u0, v1))
            # Face away from the axis: the outside of the rock mass.
            ref = sub(outer[i][j], c0)
            m.quad(a, b, c, d, ref)

    # Mouth: rim joining the interior to the outside skin, plus the far-end cap
    # of the rock mass. Both are only ever seen from outside the cave.
    mouth_t = fr[0][1]
    for j in range(RING):
        jn = (j + 1) % RING
        a = (inner[0][j], (0.0, j / float(RING) * 4.0))
        b = (outer[0][j], (1.0, j / float(RING) * 4.0))
        c = (outer[0][jn], (1.0, (j + 1) / float(RING) * 4.0))
        d = (inner[0][jn], (0.0, (j + 1) / float(RING) * 4.0))
        m.quad(a, b, c, d, mul(mouth_t, -1.0))

    for j in range(RING):
        jn = (j + 1) % RING
        a = (outer[-1][j], (0.0, j / float(RING) * 4.0))
        b = (outer[-1][jn], (1.0, (j + 1) / float(RING) * 4.0))
        c = (last_center, (0.5, 0.5))
        m.tri(a, b, c, fr[-1][1])

    m.cut(roof_hole)
    return m


def shaft_mesh():
    """Daylight shaft over the chamber: inner wall, rock mass, top rim."""
    inner, outer = [], []
    for z, r in SHAFT:
        pts_a, pts_b = [], []
        for j in range(RING):
            theta = TAU * j / RING
            ri = radius_at(r, z * 0.7, theta)
            ro = radius_at(r + SHAFT_THICKNESS, z * 0.7 + 17.0, theta + 0.21)
            pts_a.append((SHAFT_X + ri * math.cos(theta),
                          SHAFT_Y + ri * math.sin(theta), z))
            pts_b.append((SHAFT_X + ro * math.cos(theta),
                          SHAFT_Y + ro * math.sin(theta), z))
        inner.append(pts_a)
        outer.append(pts_b)

    m = Mesh()
    for i in range(len(SHAFT) - 1):
        for j in range(RING):
            jn = (j + 1) % RING
            u0, u1 = SHAFT[i][0] / 3.0, SHAFT[i + 1][0] / 3.0
            v0, v1 = j / float(RING) * 3.0, (j + 1) / float(RING) * 3.0
            a = (inner[i][j], (u0, v0))
            b = (inner[i + 1][j], (u1, v0))
            c = (inner[i + 1][jn], (u1, v1))
            d = (inner[i][jn], (u0, v1))
            ref = sub((SHAFT_X, SHAFT_Y, inner[i][j][2]), inner[i][j])
            m.quad(a, b, c, d, ref)

            a = (outer[i][j], (u0, v0))
            b = (outer[i + 1][j], (u1, v0))
            c = (outer[i + 1][jn], (u1, v1))
            d = (outer[i][jn], (u0, v1))
            ref = sub(outer[i][j], (SHAFT_X, SHAFT_Y, outer[i][j][2]))
            m.quad(a, b, c, d, ref)

    # Rim at the open top of the shaft.
    for j in range(RING):
        jn = (j + 1) % RING
        a = (inner[-1][j], (0.0, j / float(RING) * 3.0))
        b = (outer[-1][j], (1.0, j / float(RING) * 3.0))
        c = (outer[-1][jn], (1.0, (j + 1) / float(RING) * 3.0))
        d = (inner[-1][jn], (0.0, (j + 1) / float(RING) * 3.0))
        m.quad(a, b, c, d, (0.0, 0.0, 1.0))
    return m


def combine(*meshes):
    out = Mesh()
    for m in meshes:
        out.tris.extend(m.tris)
    return out


# ---------------------------------------------------------------------------
# Stalactites and stalagmites
# ---------------------------------------------------------------------------
# (distance along the centreline, angle around the ring in degrees, length,
# base radius). Angles near 90 hang from the roof, near 270 stand on the floor.
# Nothing is placed across the squeeze (s 34..38) or in the roof directly under
# the daylight shaft, so neither the chamber entrance nor the light is blocked;
# the tall mouth zone and gallery get the big formations.
SPELEOTHEMS = [
    # mouth zone, where the roof is at its highest
    (2.6, 96, 1.75, 0.42), (3.4, 74, 1.15, 0.30), (5.2, 112, 0.95, 0.26),
    (6.4, 88, 1.45, 0.34), (7.8, 62, 0.80, 0.22), (9.1, 104, 1.25, 0.30),
    (10.4, 80, 0.70, 0.20), (11.9, 118, 0.90, 0.24),
    # gallery
    (15.2, 92, 1.55, 0.36), (16.4, 70, 0.85, 0.24), (18.0, 108, 1.30, 0.32),
    (19.6, 84, 0.95, 0.26), (21.8, 100, 1.60, 0.38), (23.2, 66, 0.75, 0.20),
    (25.0, 94, 1.20, 0.30), (26.6, 114, 0.85, 0.22), (28.4, 78, 1.05, 0.26),
    (30.6, 98, 0.70, 0.20), (32.4, 86, 0.55, 0.16),
    # chamber, small and kept away from the shaft
    (41.0, 46, 0.45, 0.14), (42.2, 132, 0.30, 0.10),
    # formations growing up off the floor
    (3.0, 264, 1.30, 0.40), (4.2, 292, 0.85, 0.26), (5.8, 268, 0.95, 0.28),
    (7.2, 300, 0.65, 0.20), (8.6, 276, 1.55, 0.42), (10.2, 288, 0.55, 0.18),
    (12.4, 262, 0.90, 0.26), (14.6, 284, 1.10, 0.30), (17.2, 268, 0.70, 0.22),
    (19.0, 296, 0.95, 0.26), (22.4, 274, 1.35, 0.36), (24.6, 262, 0.60, 0.20),
    (27.2, 290, 1.00, 0.28), (29.6, 270, 0.75, 0.22), (31.8, 282, 0.55, 0.18),
    (40.6, 238, 0.35, 0.12),
]


def speleothem_bases():
    """Where each formation meets the rock: (x, y, z, grows_upwards)."""
    inner, _, s_values, _, _ = build_tunnel()
    out = []
    for s, angle, _, _ in SPELEOTHEMS:
        i = min(range(len(s_values)), key=lambda k: abs(s_values[k] - s))
        j = int(round(math.radians(angle) / TAU * RING)) % RING
        p = inner[i][j]
        out.append((p[0], p[1], p[2], math.sin(math.radians(angle)) < 0.0))
    return out


def add_speleothem(mesh, base, upward, length, base_r, phase):
    """Tapered spike from ::base:: down from the roof or up from the floor."""
    axis = (0.0, 0.0, 1.0 if upward else -1.0)
    levels = [(0.0, 1.0), (0.30, 0.74), (0.58, 0.46), (0.80, 0.22)]
    seg = 9
    rings, centers = [], []
    for t, rf in levels:
        c = add(base, mul(axis, length * t))
        centers.append(c)
        pts = []
        for k in range(seg):
            phi = TAU * k / seg
            rr = base_r * rf * (1.0 + 0.16 * math.sin(3.0 * phi + phase)
                                + 0.09 * math.sin(5.0 * phi - 1.7 * phase))
            lean = 0.10 * length * t
            pts.append(add(c, (math.cos(phi) * rr + lean * math.cos(phase + 1.0),
                               math.sin(phi) * rr + lean * math.sin(phase + 1.0),
                               0.0)))
        rings.append(pts)
    tip = add(base, mul(axis, length))

    for i in range(len(rings) - 1):
        mid = mul(add(centers[i], centers[i + 1]), 0.5)
        for k in range(seg):
            kn = (k + 1) % seg
            v0, v1 = rings[i][k], rings[i + 1][k]
            v2, v3 = rings[i + 1][kn], rings[i][kn]
            v0s = levels[i][0] * length / 1.5
            v1s = levels[i + 1][0] * length / 1.5
            a = (v0, (k / seg * 2.0, v0s))
            b = (v1, (k / seg * 2.0, v1s))
            c = (v2, ((k + 1) / seg * 2.0, v1s))
            d = (v3, ((k + 1) / seg * 2.0, v0s))
            mesh.quad(a, b, c, d, sub(mul(add(v0, v2), 0.5), mid))

    last = len(rings) - 1
    for k in range(seg):
        kn = (k + 1) % seg
        a = (rings[last][k], (k / seg * 2.0, levels[last][0] * length / 1.5))
        b = (rings[last][kn], ((k + 1) / seg * 2.0, levels[last][0] * length / 1.5))
        mesh.tri(a, b, (tip, (0.5, length / 1.5)),
                 sub(add(rings[last][k], rings[last][kn]), mul(centers[last], 2.0)))


def speleothem_mesh():
    inner, _, s_values, _, _ = build_tunnel()
    m = Mesh()
    for idx, (s, angle, length, base_r) in enumerate(SPELEOTHEMS):
        i = min(range(len(s_values)), key=lambda k: abs(s_values[k] - s))
        j = int(round(math.radians(angle) / TAU * RING)) % RING
        surface = inner[i][j]
        upward = math.sin(math.radians(angle)) < 0.0
        # Bury the base 0.12 m in the rock so the open base ring never shows.
        base = add(surface, (0.0, 0.0, -0.12 if upward else 0.12))
        add_speleothem(m, base, upward, length, base_r, 0.9 * idx + 0.4)
    return m


# ---------------------------------------------------------------------------
# Diagnostics: floor heights for prop placement
# ---------------------------------------------------------------------------
def floor_table():
    stations, _ = sample_path(TUNNEL, STEP)
    fr = frames([p for p, _ in stations])
    s = 0.0
    rows = []
    for i, ((p, r), (c, t, a_axis, b_axis)) in enumerate(zip(stations, fr)):
        if i > 0:
            s += math.dist(p, stations[i - 1][0])
        low = 1e9
        for j in range(RING):
            theta = TAU * j / RING
            rr = radius_at(r, s, theta)
            bl = rr * math.sin(theta)
            f = floor_depth(r)
            if f > 0.0 and bl < -f:
                bl = -(f + (-bl - f) * 0.15)
            if bl < low:
                low = bl
        rows.append((p[0], p[1], p[2] + low, r))
    return rows


def nearest_station(rows, x, y):
    best, bd = None, 1e9
    for row in rows:
        d = (row[0] - x) ** 2 + (row[1] - y) ** 2
        if d < bd:
            best, bd = row, d
    return best, math.sqrt(bd)


def main():
    visual = combine(tunnel_mesh(), shaft_mesh(), speleothem_mesh())

    os.makedirs(os.path.join(HERE, "meshes"), exist_ok=True)
    nv, nt = visual.write_dae(os.path.join(HERE, "meshes", "cave_shell.dae"),
                              "cave_shell", "cave_rock")
    print("cave_shell.dae     %6d verts %6d tris" % (nv, nt))
    roofs = sum(1 for _, a, _, _ in SPELEOTHEMS if math.sin(math.radians(a)) >= 0.0)
    print("speleothems        %d (%d stalactites, %d stalagmites)"
          % (len(SPELEOTHEMS), roofs, len(SPELEOTHEMS) - roofs))

    stations, total = sample_path(TUNNEL, STEP)
    print("centreline length  %.1f m, %d stations" % (total, len(stations)))
    print("radii              min %.2f m max %.2f m"
          % (min(r for _, r in stations), max(r for _, r in stations)))

    # Squeeze clearance: the widest circle that is guaranteed to fit anywhere
    # along it (2 x the smallest radius), and the vertical opening at the
    # tightest station (roof above floor).
    guaranteed, vertical = 1e9, 1e9
    s = 0.0
    for i, (p, r) in enumerate(stations):
        if i:
            s += math.dist(p, stations[i - 1][0])
        if 0.0 < p[0] < 2.6:
            lows, highs = [], []
            for j in range(360):
                theta = TAU * j / 360.0
                rr = radius_at(r, s, theta)
                lows.append(rr)
                bl = rr * math.sin(theta)
                f = floor_depth(r)
                if f > 0.0 and bl < -f:
                    bl = -(f + (-bl - f) * 0.15)
                highs.append(bl)
            guaranteed = min(guaranteed, 2.0 * min(lows))
            vertical = min(vertical, max(highs) - min(highs))
    print("squeeze clearance  %.2f m guaranteed circle, %.2f m roof to floor"
          % (guaranteed, vertical))

    rows = floor_table()
    print("\nfloor z by position (blank = outside the cave or far from it):")
    ys = [y * 1.0 for y in range(-8, 11, 2)]
    print("        " + "".join("%7.0f" % y for y in ys))
    for x in range(-30, 12, 2):
        line = "%6.0f  " % x
        for y in ys:
            row, d = nearest_station(rows, x, y)
            if d < row[3] * 0.85:
                line += "%+7.2f" % row[2]
            else:
                line += "      ."
        print(line)


if __name__ == "__main__":
    main()
