#!/usr/bin/env python3
"""GMT relief maps and MATLAB-binned histograms for frozen CSPP training data."""
import io
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pygmt

from prepare_training_distribution import ROOT, OUTPUT, artifact

DESTINATION = ROOT / "output/pdf"
WIDTH = 6.3
HIST_WIDTH = 4.4
HIST_HEIGHT = 5.3
BLOCK_STEP = 9.0
RED = "#cc3038"
OCEAN = "#98bbd5"
COLORS = ["#73ab72", "#a9c877", "#ddd979", "#e8da96", "#ccb995", "#b4a28d", "#a39b92", "#e7e5df"]


class Canvas:
    def __init__(self):
        self.figure = pygmt.Figure()
        self.figure.shift_origin(xshift="2c", yshift="22c")
        self.x = self.y = 0.0

    def origin(self, x, y):
        self.figure.shift_origin(xshift=f"{x-self.x}c", yshift=f"{y-self.y}c")
        self.x, self.y = x, y


def map_region(summary):
    xmin, xmax = summary["longitude_range"]
    ymin, ymax = summary["latitude_range"]
    mercator = lambda lat: math.log(math.tan(math.pi/4+math.radians(lat)/2))
    reverse = lambda value: math.degrees(2*math.atan(math.exp(value))-math.pi/2)
    xmid = math.radians((xmin+xmax)/2)
    ymid = (mercator(ymin)+mercator(ymax))/2
    span = max(math.radians(xmax-xmin), mercator(ymax)-mercator(ymin))*1.19
    return [math.degrees(xmid-span/2), math.degrees(xmid+span/2), reverse(ymid-span/2), reverse(ymid+span/2)]


def prepare_maps(summaries):
    folder = OUTPUT / "relief"
    folder.mkdir(exist_ok=True)
    files, specs = {}, []
    for summary in summaries:
        slug = summary["slug"]
        region = map_region(summary)
        grid_region = [math.floor(region[0]), math.ceil(region[1]), math.floor(region[2]), math.ceil(region[3])]
        full = folder / f"{slug}_15s.nc"
        tiles = []
        for longitude in range(math.floor(grid_region[0]/10)*10, math.ceil(grid_region[1]/10)*10, 10):
            for latitude in range(math.floor(grid_region[2]/10)*10, math.ceil(grid_region[3]/10)*10, 10):
                name = f"{'N' if latitude>=0 else 'S'}{abs(latitude):02}{'E' if longitude>=0 else 'W'}{abs(longitude):03}.earth_relief_15s_p.nc"
                path = Path.home() / ".gmt/server/earth/earth_relief/earth_relief_15s_p" / name
                assert path.is_file(), path
                tiles.append(artifact(path))
        if not full.exists():
            subprocess.run(["/opt/homebrew/bin/gmt", "grdblend", *[t["path"] for t in tiles],
                            "-G"+str(full), "-R"+"/".join(map(str, grid_region)), "-I15s", "-rp", "-Vw"], check=True)
        files[slug] = full
        specs.append({"dataset": summary["dataset"], "region": region, "grid_region": grid_region,
                      "grid": artifact(full), "source_tiles": tiles,
                      "source": "Cached GMT Earth Relief 15-arcsecond pixel-registered tiles",
                      "documentation": "https://www.generic-mapping-tools.org/remote-datasets/earth-relief.html"})
        print(f"relief ready: {summary['dataset']}", flush=True)
    (OUTPUT / "relief_sources.json").write_text(json.dumps(specs, indent=2)+"\n")
    return files


def nice_limit(maximum):
    target = maximum*1.12
    magnitude = 10**math.floor(math.log10(target))
    return next(value*magnitude for value in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if value*magnitude >= target)


def histogram(canvas, summary, field, x, y):
    canvas.origin(x, y)
    fig = canvas.figure
    stats = summary[field]
    values = np.loadtxt(OUTPUT / f"{summary['slug']}_{field}_histogram.txt")
    maximum = nice_limit(float(values[:, 1].max()))
    region = [*stats["x_limits"], 0, maximum]
    if field == "magnitude":
        title, xlabel, color, xtick = "Magnitude", "Catalog magnitude", "#b7dce4", "xa1f0.2"
        mean_label = f"Mean: {stats['mean']:.2f}"
    else:
        title, xlabel, color = "Source depth", "Depth (km)", "#afe2a3"
        xtick = "xa100f20" if region[1] > 300 else "xa40f10" if region[1] > 140 else "xa20f10"
        mean_label = f"Mean: {stats['mean']:.1f} km"
    fig.basemap(region=region, projection=f"X{HIST_WIDTH}c/{HIST_HEIGHT}c",
                frame=[f"WSne+t{title}", xtick+f"+l{xlabel}", "yag+lRecords"])
    fig.plot(x=values[:, 0], y=values[:, 1], style=f"b{stats['bin_width']}u+b0", fill=color, pen="0.45p,#596166")
    fig.plot(x=[stats["mean"]]*2, y=[0, maximum*.98], pen=f"1p,{RED},--")
    fig.text(x=region[1]-.045*(region[1]-region[0]), y=maximum*.92, text=mean_label,
             justify="TR", font="8.5p,Helvetica-Bold,#252a2e", fill="white", pen="0.35p,#92999c", clearance="0.06c")


def block(canvas, summary, letter, offset, grid):
    fig = canvas.figure
    canvas.origin(0, offset)
    fig.text(x=0, y=7.18, text=f"({letter})  {summary['dataset']}", justify="BL",
             region=[0, 18, 0, 8], projection="X18c/8c", font="13p,Helvetica-Bold,#171b1e", no_clip=True)
    fig.text(x=0, y=6.67, text=f"{summary['record_count']:,} training records; {summary['event_count']:,} earthquakes",
             justify="BL", font="9p,Helvetica,#3f484e", no_clip=True)
    region = map_region(summary)
    projection = f"M{WIDTH}c"
    pygmt.makecpt(cmap=",".join(COLORS), series="0/5000/8+n", continuous=True, overrule_bg=True)
    fig.grdimage(grid=str(grid), region=region, projection=projection, cmap=True, dpi=400,
                 shading="+a315+nt0.18", frame=False)
    fig.coast(resolution="i", shorelines="0.45p,#39464c", borders="1/0.25p,#7b8286")
    fig.grdcontour(grid=str(grid), levels=1000, annotation="-", limit=[500, 5000], pen="0.12p,#b1ad96")
    points = np.loadtxt(OUTPUT / f"{summary['slug']}_epicenters.txt")
    fig.plot(x=points[:, 0], y=points[:, 1], style="t0.10c", fill=RED, pen="0.15p,#92242b")
    span = region[1]-region[0]
    tick = 4 if span > 15 else 2 if span > 8 else 1
    fig.basemap(frame=["WSne", f"xa{tick}f+lLongitude", f"ya{tick}f+lLatitude"])
    spec = io.StringIO(f"S 0.13c t 0.13c {RED} 0.2p,#92242b 0.33c Epicenters (n={summary['mapped_events']})\n")
    corner = "TR" if summary["slug"] == "instancegm" else "TL"
    x_points = (points[:, 0]-region[0])/(region[1]-region[0])*WIDTH
    mercator = lambda lat: np.log(np.tan(np.pi/4+np.radians(lat)/2))
    y_points = (mercator(points[:, 1])-mercator(region[2]))/(mercator(region[3])-mercator(region[2]))*WIDTH
    left, right = (3.2, 6.18) if corner == "TR" else (.12, 3.1)
    assert not np.any((x_points > left-.07) & (x_points < right+.07) & (y_points > 5.55))
    with pygmt.config(FONT_ANNOT_PRIMARY="7.5p,Helvetica,#242b30"):
        fig.legend(spec=spec, position=f"j{corner}+o0.12c/0.12c", box="+gwhite+p0.3p,#8e989e")
    with pygmt.config(FONT_ANNOT_PRIMARY="11.7p,Helvetica,#242b30", FONT_LABEL="12.5p,Helvetica,#242b30"):
        fig.colorbar(position="jBC+w5.4c/0.22c+h+o0c/-0.7c+m", frame=["xa2000f1000+lElevation (m)"])
    histogram(canvas, summary, "magnitude", 7.6, offset+.45)
    histogram(canvas, summary, "depth_km", 13.5, offset+.45)


def main():
    summary = json.loads((OUTPUT / "distribution_summary.json").read_text())
    assert summary["metadata_conflicts"] == 0
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with pygmt.config(PROJ_ELLIPSOID="Sphere", MAP_FRAME_TYPE="plain",
                      FONT_ANNOT_PRIMARY="8.5p,Helvetica,#242b30", FONT_LABEL="9p,Helvetica,#242b30",
                      FONT_TITLE="10p,Helvetica-Bold,#242b30", MAP_TITLE_OFFSET="0.15c",
                      MAP_LABEL_OFFSET="0.12c", MAP_ANNOT_OFFSET_PRIMARY="0.08c",
                      MAP_TICK_LENGTH_PRIMARY="0.10c", MAP_FRAME_PEN="0.6p,#41494e",
                      MAP_GRID_PEN_PRIMARY="0.2p,#e5e9eb", FORMAT_GEO_MAP="dddF",
                      COLOR_NAN="white", COLOR_BACKGROUND=OCEAN, COLOR_FOREGROUND=COLORS[-1],
                      PS_MEDIA="A0", GMT_VERBOSE="w"):
        grids = prepare_maps(summary["datasets"])
        outputs = []
        for index, dataset in enumerate(summary["datasets"]):
            canvas = Canvas()
            block(canvas, dataset, "abc"[index], 0, grids[dataset["slug"]])
            filename = DESTINATION / f"cspp_training_distribution_{dataset['slug']}.png"
            canvas.figure.savefig(str(filename), dpi=400, anti_alias=True)
            outputs.append(artifact(filename))
            print(f"rendered {dataset['dataset']}", flush=True)
        canvas = Canvas()
        for index, dataset in enumerate(summary["datasets"]):
            block(canvas, dataset, "abc"[index], -index*BLOCK_STEP, grids[dataset["slug"]])
        canvas.origin(0, -2*BLOCK_STEP)
        canvas.figure.text(x=0, y=-1.75, text="Maps: unique earthquakes. Histograms: selected training records. Dashed lines: record-weighted means.",
                           region=[0, 18, 0, 8], projection="X18c/8c", justify="BL", font="8p,Helvetica,#424c52", no_clip=True)
        for suffix in ("pdf", "png"):
            filename = DESTINATION / f"cspp_training_distribution_all.{suffix}"
            canvas.figure.savefig(str(filename), dpi=400, anti_alias=True)
            outputs.append(artifact(filename))
        result = {"outputs": outputs, "pygmt_version": pygmt.__version__,
                  "gmt_version": subprocess.check_output(["/opt/homebrew/bin/gmt", "--version"], text=True).strip(),
                  "source": artifact(Path(__file__)), "summary": artifact(OUTPUT / "distribution_summary.json"),
                  "pdf_content": "Vector symbols, bars, axes and text over raster topographic relief"}
        (OUTPUT / "figure_manifest.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
