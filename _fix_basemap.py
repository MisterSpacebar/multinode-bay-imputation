import pathlib, re

files = [
    'map_viz.py', 'map_viz_extended.py', 'canal_stress_map.py',
    'canal_stress_map_v2.py', 'dieoff_analysis.py', 'paper_figure.py',
]
NEW_CALL = 'ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.Esri.WorldGrayCanvas, zoom=12, attribution=False)'

for fn in files:
    p = pathlib.Path(fn)
    txt = p.read_text(encoding='utf-8')
    new = txt.replace('ax.set_facecolor("#d4e9f7")', NEW_CALL)
    new = new.replace("ax.set_facecolor('#d4e9f7')", NEW_CALL)
    new = new.replace('ax.set_facecolor("#e8f4f8")', NEW_CALL)
    new = new.replace("ax.set_facecolor('#e8f4f8')", NEW_CALL)
    if new != txt:
        p.write_text(new, encoding='utf-8')
        print(fn, '- updated')
    else:
        print(fn, '- no change (check manually)')
