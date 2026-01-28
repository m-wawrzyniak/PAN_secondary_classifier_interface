import os
import pandas as pd
from pathlib import Path
import math

def export_html_paginated(name, csv_path, extracted_path, page_size=100):
    """
    Creates paginated HTML visualization for is_face / non_face images.

    Parameters
    ----------
    csv_path : str or Path
        Path to manual_class.csv
    page_size : int
        How many images per HTML page (default 100)
    """

    csv_path = Path(csv_path)
    base_dir = csv_path.parent
    title = base_dir.name

    df = pd.read_csv(csv_path)
    face_df = df[df["is_face"] == 1].reset_index(drop=True)
    nonface_df = df[df["is_face"] == 0].reset_index(drop=True)

    def write_page(df_subset, label, page_idx, total_pages):
        """Write a single paginated HTML file."""
        file_name = f"{name}_{label.lower()}_page_{page_idx+1}.html"
        os.makedirs(base_dir / "result_html", exist_ok=True)
        out_path = base_dir / "result_html" /file_name

        start = page_idx * page_size
        end = min(start + page_size, len(df_subset))
        subset = df_subset.iloc[start:end]

        # Navigation links
        nav = "<div style='margin-bottom:10px;'>"
        if page_idx > 0:
            prev_page = f"{name}_{label.lower()}_page_{page_idx}.html"
            nav += f"<a href='{prev_page}'>⬅️ Prev</a> | "
        nav += f"Page {page_idx + 1} / {total_pages}"
        if page_idx < total_pages - 1:
            next_page = f"{name}_{label.lower()}_page_{page_idx + 2}.html"
            nav += f" | <a href='{next_page}'>Next ➡️</a>"
        nav += "</div><hr>"

        html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title} — {label} {page_idx+1}/{total_pages}</title>
</head>
<body style="font-family:sans-serif;">
<h1>{label} — {title}</h1>
{nav}
<div style='display:flex; flex-wrap:wrap;'>
"""

        for fname in subset["frame"]:
            img_path = extracted_path / fname
            if not img_path.exists():
                continue

            html += (
                f"<div style='margin:10px;'>"
                f"<img loading='lazy' src='{img_path}' "
                f"style='max-width:320px;height:auto;display:block;'>"
                f"<p style='text-align:center;'>{fname}</p>"
                f"</div>"
            )

        html += f"</div>{nav}</body></html>"

        out_path.write_text(html, encoding="utf-8")

    def paginated_export(df_subset, label):
        if len(df_subset) == 0:
            return []

        total_pages = math.ceil(len(df_subset) / page_size)
        for page_idx in range(total_pages):
            write_page(df_subset, label, page_idx, total_pages)

        return [
            f"{name}_{label.lower()}_page_{i + 1}.html"
            for i in range(total_pages)
        ]

    # Generate pages
    face_pages = paginated_export(face_df, "Faces")
    nonface_pages = paginated_export(nonface_df, "NonFaces")

    # Create index.html as a hub
    index_path = base_dir / "result_html" / f"{name}_index.html"
    html_index = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title} — Overview</title>
</head>
<body style="font-family:sans-serif;">
<h1>Manual Classification Overview — {title}</h1>

<h2>Faces ({len(face_df)})</h2>
<ul>
    {''.join(f"<li><a href='{p}'>{p}</a></li>" for p in face_pages)}
</ul>

<h2>Non-Faces ({len(nonface_df)})</h2>
<ul>
    {''.join(f"<li><a href='{p}'>{p}</a></li>" for p in nonface_pages)}
</ul>

</body>
</html>
"""
    index_path.write_text(html_index, encoding="utf-8")
    print(f"\t\tOK Created HTML viewer for {name}")
