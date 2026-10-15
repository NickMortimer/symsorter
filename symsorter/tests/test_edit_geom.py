"""Tests for outline edit geometry (fit zoom, lasso delete, eraser, insert)."""

from __future__ import annotations

from symsorter.outline.edit_geom import (
    delete_points_inside,
    erase_points_near,
    fit_view,
    insert_vertex,
    nearest_edge,
    nearest_vertex,
    padded_bbox,
    polygons_bbox,
    zoom_at_pointer,
)


def test_polygons_bbox_and_pad() -> None:
    polys = [
        {"points": [[10.0, 20.0], [50.0, 20.0], [50.0, 60.0], [10.0, 60.0]]},
    ]
    bbox = polygons_bbox(polys)
    assert bbox == (10.0, 20.0, 50.0, 60.0)
    padded = padded_bbox(bbox, pad_frac=0.10, img_w=224, img_h=224)
    # 10% of 40×40 → 4px pad
    assert padded == (6.0, 16.0, 54.0, 64.0)


def test_fit_view_zooms_to_polys() -> None:
    polys = [
        {"points": [[40.0, 40.0], [80.0, 40.0], [80.0, 80.0], [40.0, 80.0]]},
    ]
    z, ox, oy = fit_view(
        view_w=400, view_h=400, img_w=224, img_h=224, polygons=polys, pad_frac=0.10
    )
    # Should be much larger than fitting the whole 224 image into 400
    z_full, _, _ = fit_view(
        view_w=400, view_h=400, img_w=224, img_h=224, polygons=[], pad_frac=0.10
    )
    assert z > z_full
    assert z > 1.0


def test_zoom_at_pointer_keeps_focus() -> None:
    zoom, ox, oy = 2.0, 10.0, 20.0
    px, py = 100.0, 150.0
    ix = (px - ox) / zoom
    iy = (py - oy) / zoom
    nz, nox, noy = zoom_at_pointer(
        zoom=zoom, offset_x=ox, offset_y=oy, pointer_x=px, pointer_y=py, factor=2.0
    )
    assert nz == 4.0
    assert abs((px - nox) / nz - ix) < 1e-9
    assert abs((py - noy) / nz - iy) < 1e-9


def test_nearest_vertex_and_edge_insert() -> None:
    pts = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
    assert nearest_vertex(pts, 2.0, 3.0, max_dist=10.0) == 0
    assert nearest_vertex(pts, 50.0, 50.0, max_dist=5.0) is None
    edge = nearest_edge(pts, 50.0, 2.0, max_dist=10.0)
    assert edge is not None
    after, px, py = edge
    assert after == 0
    assert abs(px - 50.0) < 1e-6
    assert abs(py - 0.0) < 1e-6
    new = insert_vertex(pts, after, px, py)
    assert len(new) == 5
    assert new[1] == [50.0, 0.0]


def test_lasso_deletes_points_inside() -> None:
    pts = [
        [10.0, 10.0],
        [90.0, 10.0],
        [90.0, 90.0],
        [50.0, 50.0],  # inside lasso below
        [10.0, 90.0],
    ]
    # Lasso covering the center point
    lasso = [[40.0, 40.0], [70.0, 40.0], [70.0, 70.0], [40.0, 70.0]]
    kept, n = delete_points_inside(pts, lasso, min_keep=3)
    assert n == 1
    assert len(kept) == 4
    assert [50.0, 50.0] not in kept


def test_lasso_refuses_below_min_keep() -> None:
    pts = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
    lasso = [[-1.0, -1.0], [20.0, -1.0], [20.0, 20.0], [-1.0, 20.0]]
    kept, n = delete_points_inside(pts, lasso, min_keep=3)
    assert n == 0
    assert kept == pts


def test_eraser_removes_nearby() -> None:
    pts = [[0.0, 0.0], [50.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
    kept, n = erase_points_near(pts, 50.0, 1.0, radius=5.0, min_keep=3)
    assert n == 1
    assert [50.0, 0.0] not in kept
    assert len(kept) == 4
